--[[

--------------------------------------------------
Recurrent Neural Networks for Tweet Classification
--------------------------------------------------
Default setting for RNN is obtained from [1].

References:
1. https://github.com/stanfordnlp/treelstm

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
require 'nngraph'
require 'xlua'
require 'dpnn'
require 'rnn'
tds = require('tds')
local utils = require 'utils'

cmd = torch.CmdLine()

cmd:option('-data', 'data/', 'data folder')
cmd:option("-seed", 123, 'seed for the random generator')
cmd:option('-pre_train', 0, 'initialize word embeddings with pre-trained vectors?')
cmd:option('-glove_dir', 'data/', 'Directory for accesssing the pre-trained glove word embeddings')

cmd:option('-dim', 100, ' dimensionality of word embeddings')
cmd:option('-min_freq', 1, 'words that occur less than <int> times will not be taken for training')
cmd:option('-optim_method', 'adagrad', 'Gradient descent method. Options: adadelta, adam, adagrad')
cmd:option('-num_epochs', 10, 'number of full passes through the training data')
cmd:option('-bsize', 50, 'mini-batch size')
cmd:option('-layers', '{150}', 'size of each hidden layer in RNN (ex: for 3-layer net, give it as {150,150,150})')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
cmd:option('-dropout', 0.5, 'dropout for regularization, used before the prediction layer. 0 = no dropout')
cmd:option('-rnn_type', 'rnn', 'which model to use? lstm, rnn, gru')
cmd:option('-seq_length', 50, 'no. of timessteps to unroll for')
cmd:option('-learning_rate', 0.05, 'learning rate')
cmd:option('-reg', 0.1, 'l2 regularization hyper-parameter')

params = cmd:parse(arg)
params.ZERO = '<zero_rnn>'

torch.manualSeed(params.seed)

params.optim_state = { learningRate = params.learning_rate }
params.max_words = params.seq_length

if params.optim_method == 'adadelta' then params.grad_desc = optim.adadelta
	elseif params.optim_method == 'adagrad' then params.grad_desc = optim.adagrad
		else params.grad_desc = optim.adam end

-- Function to tokenize the input tweet to words
function params.tokenizeTweet(tweet, max_words)
    local words = {}
    table.insert(words, '<sot>') -- start token
    local _words = stringx.split(tweet)
    for i = 1, #_words do
    	if max_words == nil or #words < max_words then
        	table.insert(words, _words[i])
        end
    end
    if max_words == nil or #words < max_words then
    	table.insert(words, '<eot>') -- end token
    end 
    if max_words ~= nil  then
    	local pads = max_words - #words
    	if pads > 0 then
    		local new_words = {}
    		for i = 1, pads do
    			table.insert(new_words, params.ZERO)
    		end
    		for i = 1, #words do
    			table.insert(new_words, words[i])
    		end
    		return new_words
    	end
    end
    return words
end

-- Build vocab.
utils.build_vocab(params)

-- Get train, dev and test tensors.
utils.get_tensors(params)

-- Erect the model and criterion
params.ngram_lookup = nn.LookupTableMaskZero(#params.index2word, params.dim)
params.model = nn.Sequential()
params.model:add(params.ngram_lookup)
params.model:add(nn.SplitTable(1, 2))
local layers = loadstring(" return "..params.layers)()
local input_size = params.dim
for i, hidden_size in ipairs(layers) do
	local rnn
	if params.rnn_type == 'rnn' then
		rnn = nn.Recurrent(hidden_size, nn.MaskZero(nn.Linear(input_size, hidden_size), 1), 
			nn.MaskZero(nn.Linear(hidden_size, hidden_size), 1), nn.Sigmoid(), params.max_words)
	elseif params.rnn_type == 'gru' then
		rnn = nn.GRU(input_size, hidden_size, nil, 0)
	elseif params.rnn_type == 'lstm' then
		nn.FastLSTM.usenngraph = true
		rnn = nn.FastLSTM(input_size, hidden_size)
	end
	input_size = hidden_size
	rnn = nn.Sequencer(rnn)
	params.model:add(rnn)
end
params.model:add(nn.MaskZero(nn.SelectTable(-1), 1))
if params.dropout > 0 then params.model:add(nn.MaskZero(nn.Dropout(params.dropout), 1)) end
params.model:add(nn.MaskZero(nn.Linear(input_size, #params.index2label), 1))
params.model = params.model:cuda()
params.criterion = nn.MaskZeroCriterion(nn.CrossEntropyCriterion(), 1):cuda()

params.pp, params.gp = params.model:getParameters()

if params.pre_train == 1 then
	local glove_complete_path = params.glove_dir .. 'glove.twitter.27B.' .. params.dim .. 'd.txt.gz'
	local is_present = lfs.attributes(glove_complete_path) or -1
	if is_present ~= -1 then
		utils.init_word_weights(params, params.ngram_lookup, glove_complete_path)
	else
		print('>>>WARNING>>> Specified glove embedding file is not found at: '..glove_complete_path)
	end
end

-- Function to train the model for 1 epoch
function train(optim_states)
	params.model:training()
	local indices = torch.randperm(#params.train_tensors)
	local input_x, input_y = torch.CudaTensor(params.bsize, params.max_words), torch.CudaTensor(params.bsize, 1) 
	local epoch_loss, num_batches = 0, math.floor(#params.train_tensors / params.bsize)
	function train_batch(batch_size)
		local feval = function(x)		
			params.gp:zero()
			local outputs = params.model:forward(input_x)
			local example_loss = params.criterion:forward(outputs, input_y)
			epoch_loss = epoch_loss + (example_loss * batch_size)
			local grads = params.criterion:backward(outputs, input_y)
			params.model:backward(input_x, grads)
			return example_loss, params.gp
		end
		params.grad_desc(feval, params.pp, params.optim_state)
	end
	xlua.progress(1, num_batches)
	for i = 1, num_batches do
		for j = 1, params.bsize do
			local idx = indices[(params.bsize*(i-1))+j]
			input_x[j] = params.train_tensors[idx][1]
			input_y[j] = params.train_tensors[idx][2]
		end
		train_batch(params.bsize)
		if i%10 == 0 then xlua.progress(i, num_batches) end
	end
	local rem_inputs = (#params.train_tensors - (num_batches * params.bsize))
	if rem_inputs > 0 then
		input_x, input_y = torch.CudaTensor(rem_inputs, params.max_words), torch.CudaTensor(rem_inputs, 1) 
		local start = (num_batches * params.bsize)
		for i = 1, rem_inputs do
			local idx = indices[start+i]
			input_x[i] = params.train_tensors[idx][1]
			input_y[i] = params.train_tensors[idx][2]
		end
		train_batch(rem_inputs)
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