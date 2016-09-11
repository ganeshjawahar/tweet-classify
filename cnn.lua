--[[

---------------------------------------------------------
Convolutional Neural Networks for Sentence Classification
---------------------------------------------------------
Default setting for CNN is obtained from [3].

References:
1. http://www.aclweb.org/anthology/D14-1181
2. https://github.com/yoonkim/CNN_sentence
3. https://github.com/harvardnlp/sent-conv-torch

]]--

require 'torch'
require 'nn'
require 'sys'
require 'optim'
require 'xlua'
require 'lfs'
require 'cunn'
require 'cutorch'
require 'pl.stringx'
require 'pl.file'
paths.dofile('models/kim_sent.lua')
tds = require('tds')
local utils = require 'utils'

cmd = torch.CmdLine()

cmd:option('-data', 'data/', 'data folder')
cmd:option('-seed', 123, 'seed for the random generator')
cmd:option('-pre_train', 0, 'initialize word embeddings with pre-trained vectors?')
cmd:option('-glove_dir', 'data/', 'Directory for accesssing the pre-trained glove word embeddings')

cmd:option('-dim', 200, ' dimensionality of word embeddings')
cmd:option('-min_freq', 1, 'words that occur less than <int> times will not be taken for training')
cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
cmd:option('-num_epochs', 25, 'number of full passes through the training data')
cmd:option('-bsize', 50, 'mini-batch size')
cmd:option('-L2s', 3, 'L2 normalize weights')
cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
cmd:option('-kernels', '3,4,5', 'Kernel sizes of convolutions, table format.')
cmd:option('-dropout_p', 0.5, 'p for dropout')

params = cmd:parse(arg)
params.ZERO = '<zero_cnn>'

torch.manualSeed(params.seed)

if params.optim_method == 'adadelta' then params.grad_desc = optim.adadelta 
	else params.grad_desc = optim.adam end

-- Function to tokenize the input tweet to words
function params.tokenizeTweet(tweet, max_words)
	if max_words == nil then max_words = 10000 end
    local words = {}
    for i = 1, 4 do table.insert(words, params.ZERO) end
    table.insert(words, '<sot>')
    local _words = stringx.split(tweet)
    for i = 1, #_words do
        if #words < max_words then table.insert(words, _words[i]) end
    end
    if #words < max_words then table.insert(words, '<eot>') end
    for i = 1, 4 do if #words < max_words then table.insert(words, params.ZERO) end end
    return words
end

-- Build vocab.
utils.build_vocab(params)

-- Get train, dev and test tensors.
utils.get_tensors(params)

-- Erect the model and criterion
config = {}
config.ngram_lookup_rows = #params.index2word
config.ngram_lookup_cols = params.dim
config.num_classes = #params.index2label
config.kernels = params.kernels
config.num_feat_maps = params.num_feat_maps
config.dropout_p = params.dropout_p
config.max_words = params.max_words
params.model = get_model(config)
params.model = params.model:cuda()
params.criterion = nn.CrossEntropyCriterion():cuda()

params.layers = {}
params.layers.ngram_lookup = utils.get_layer(params.model, 'ngram_lookup')
params.layers.linear_pred_layer = utils.get_layer(params.model, 'linear_pred_layer')

params.pp, params.gp = params.model:getParameters() -- flattens all the model parameters into one fat tensor
params.layers.ngram_lookup.weight:normal(-0.25, 0.25)
params.layers.linear_pred_layer.weight:normal():mul(0.01)
params.layers.linear_pred_layer.bias:zero()

params.input_tensors, params.target_tensors = torch.CudaTensor(params.bsize, params.max_words), torch.CudaTensor(params.bsize)
params.layers.ngram_lookup.weight[params.word2index[params.ZERO]]:zero()

if params.pre_train == 1 then
	local glove_complete_path = params.glove_dir .. 'glove.twitter.27B.' .. params.dim .. 'd.txt.gz'
	local is_present = lfs.attributes(glove_complete_path) or -1
	if is_present ~= -1 then
		utils.init_word_weights(params, params.layers.ngram_lookup, glove_complete_path)
	else
		print('>>>WARNING>>> Specified glove embedding file is not found at: '..glove_complete_path)
	end
end

-- Function to train the model for 1 epoch
function train(optim_states)
	params.model:training()
	local ids = torch.randperm(#params.train_tensors)
	local cur_id, epoch_loss, num_batches = 0, 0, math.floor(#params.train_tensors / params.bsize)	
	local config
	if params.optim_method == 'adadelta' then
		config = { rho = 0.95, eps = 1e-6 }
	else
		config = { }
	end
	xlua.progress(1, num_batches)
	for batch = 1, num_batches do
		local cur_count = 0
		while cur_id < #params.train_tensors and cur_count < params.bsize  do
			cur_count = cur_count + 1
			cur_id = cur_id + 1
			params.input_tensors[cur_count] = params.train_tensors[ids[cur_id]][1]
			params.target_tensors[cur_count] = params.train_tensors[ids[cur_id]][2]
		end
		local feval = function(x)
			if x ~= params.pp then
				params.pp:copy(x)
			end		
			params.gp:zero()
			local out = params.model:forward(params.input_tensors)
			local loss = params.criterion:forward(out, params.target_tensors)
			epoch_loss = epoch_loss + (loss * params.bsize)
			local grads = params.criterion:backward(out, params.target_tensors)
			params.model:backward(params.input_tensors, grads) 
			return loss, params.gp
		end
		params.grad_desc(feval, params.pp, config, optim_states)

		-- reset padding embedding to zero
		params.layers.ngram_lookup.weight[params.word2index[params.ZERO]]:zero()
		-- Renorm (Euclidean projection to L2 ball)
    	local renorm = function(row)
    		local n = row:norm()
      		row:mul(params.L2s):div(1e-7 + n)
    	end
	    -- renormalize linear row weights
    	local w = params.layers.linear_pred_layer.weight
    	for j = 1, w:size(1) do
      		renorm(w[j])
    	end
    	if batch%10 == 0 then xlua.progress(batch, num_batches) end
	end
	local rem_samples = #params.train_tensors - cur_id
	if rem_samples > 0 then
		local sub_train_input_tensors, sub_train_target_tensors = torch.CudaTensor(rem_samples, params.max_words), torch.CudaTensor(rem_samples)
		for i = 1, rem_samples do
			cur_id = cur_id + 1
			sub_train_input_tensors[i] = params.train_tensors[ids[cur_id]][1]
			sub_train_target_tensors[i] = params.train_tensors[ids[cur_id]][2]
		end
		local feval = function(x)
			if x ~= params.pp then
				params.pp:copy(x)
			end		
			params.gp:zero()
			local out = params.model:forward(sub_train_input_tensors)
			local loss = params.criterion:forward(out, sub_train_target_tensors)
			epoch_loss = epoch_loss + (loss * rem_samples)
			local grads = params.criterion:backward(out, sub_train_target_tensors)
			params.model:backward(sub_train_input_tensors, grads) 
			return loss, params.gp
		end
		params.grad_desc(feval, params.pp, config, optim_states)
		params.layers.ngram_lookup.weight[params.word2index[params.ZERO]]:zero()
		local renorm = function(row)
    		local n = row:norm()
      		row:mul(params.L2s):div(1e-7 + n)
    	end
    	local w = params.layers.linear_pred_layer.weight
    	for j = 1, w:size(1) do
      		renorm(w[j])
    	end
	end
	xlua.progress(num_batches, num_batches)
	return (epoch_loss / #params.train_tensors)
end

-- get 2D input and target tensors.
function stackSamples(x_tensors)
	local input, target = torch.CudaTensor(#x_tensors, params.max_words), {}
	for i = 1, #x_tensors do
		input[i] = x_tensors[i][1]
		table.insert(target, x_tensors[i][2])
	end
	return input, target
end
params.val_in, params.val_tar = stackSamples(params.val_tensors)
params.test_in, params.test_tar = stackSamples(params.test_tensors)

-- get f_score and accuracy.
params.soft_max = nn.SoftMax():cuda()
function compute_performance(x_in, x_tar, model)
	model:evaluate()
	local out = model:forward(x_in)
	local pred_fi = params.soft_max:forward(out)
	local tp, pred_as, gold_as, fscores = {}, {}, {}, torch.Tensor(#params.index2label)
	for i = 1, #params.index2label do
		tp[i] = 0
		pred_as[i] = 0
		gold_as[i] = 0
	end
	local accuracy = 0
	for i = 1, #x_tar do
		local pred_label = 1
		for j = 2, #params.index2label do
			if pred_fi[i][j] > pred_fi[i][pred_label] then
				pred_label = j
			end
		end
		if pred_label == x_tar[i] then 
			tp[pred_label] = tp[pred_label] + 1 
			accuracy = accuracy + 1
		end
		pred_as[pred_label] = pred_as[pred_label] + 1
		gold_as[x_tar[i]] = gold_as[x_tar[i]] + 1
	end
	local tp_sum, prec_den, recall_den = 0, 0, 0
	for i = 1, #params.index2label do
		tp_sum = tp_sum + tp[i]
		prec_den = prec_den + pred_as[i]
		recall_den = recall_den + gold_as[i]
	end
	local micro_prec, micro_recall = (tp_sum / prec_den), (tp_sum / recall_den)
	return ((2 * micro_prec * micro_recall) / (micro_prec + micro_recall)), (accuracy / #x_tar)
end

-- Training.
-- Gradient descent state should persist over epochs
print('training...')
local optim_states = {}
local best_acc, best_fscore, best_acc_model, best_fscore_model, best_acc_epoch, best_fscore_epoch = -1, -1, nil, nil, -1, -1
for epoch = 1, params.num_epochs do
	local epoch_start = sys.clock()
	local loss = train(optim_states)
	local cur_fscore, cur_acc = compute_performance(params.val_in, params.val_tar, params.model)
	if best_fscore < cur_fscore then 
		best_fscore = cur_fscore
		best_fscore_model = params.model:clone()
		best_fscore_epoch = epoch
	end
	if best_acc < cur_acc then
		best_acc = cur_acc
		best_acc_model = params.model:clone()
		best_acc_epoch = epoch
	end
	print('Epoch ('..epoch..'/'..params.num_epochs..') Loss = '..loss..'; best_fscore = '..best_fscore..'; best_acc = '..best_acc..'; Time = '..((sys.clock() - epoch_start)/60)..' min')
end

-- Evaluation
local final_fscore, _ = compute_performance(params.test_in, params.test_tar, best_fscore_model)
local _, final_acc = compute_performance(params.test_in, params.test_tar, best_acc_model)
print('Final Fscore = '..final_fscore..' ('..best_fscore_epoch..'); Acc = '..final_acc..' ('..best_fscore_epoch..');')