--[[

-- misc utilities common to all the models

]]--

local utils={}

-- Function to build vocabulary (word-index maps)
function utils.build_vocab(params)
	print('Building vocabulary...')
	local start = sys.clock()
	
	params.vocab = tds.hash()
	params.index2word = tds.hash()
	params.word2index = tds.hash()
	params.tweet_count, params.word_count, params.unique_words, params.empty_tweets, params.max_words = 0, 0, 0, 0, 0
	local fptr = io.open(params.data..'/train.tsv', 'r')
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local tweet = stringx.split(line, '\t')[1]
		if tweet ~= nil then
			local words = params.tokenizeTweet(tweet)
			for j = 1, #words do
				local word = words[j]
				if params.vocab[word] == nil then
					params.vocab[word] = 1
					params.unique_words = params.unique_words + 1
				else
					params.vocab[word] = params.vocab[word] + 1
				end
				params.word_count = params.word_count + 1
			end
			if #words > params.max_words then
				params.max_words = #words
			end
		else
			params.empty_tweets = params.empty_tweets + 1
		end
		params.tweet_count = params.tweet_count + 1
	end
	io.close(fptr)

	for word, count in pairs(params.vocab) do
		if count < params.min_freq then
			params.vocab[word] = nil
		else
			params.index2word[#params.index2word + 1] = word
			params.word2index[word] = #params.index2word
		end
	end

	params.UK = '<UK>'
	params.vocab[params.UK] = 1
	params.index2word[#params.index2word + 1] = params.UK
	params.word2index[params.UK] = #params.index2word

	params.vocab[params.ZERO] = 1
	params.index2word[#params.index2word + 1] = params.ZERO
	params.word2index[params.ZERO] = #params.index2word
	params.vocab_size = #params.index2word
	params.unique_words = params.unique_words + 1

	print(string.format("%d (%d) (l-%d) words, %d (%d) tweets processed in %.2f minutes.", params.word_count, params.unique_words, params.max_words, params.tweet_count, params.empty_tweets, (sys.clock() - start)/60))
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d", params.min_freq, params.vocab_size))
end

function _get_tensors(params, file)
	local fptr = io.open(file, 'r')
	local tensors = {}
	local index2label, label2index = {}, {}
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local content = stringx.split(line, '\t')
		local unigrams = params.tokenizeTweet(content[1], params.max_words)
		local label = content[2]
		if label2index[label] == nil then
			index2label[#index2label + 1] = label
			label2index[label] = #index2label
		end
		local tensor = torch.CudaTensor(params.max_words):fill(params.word2index[params.ZERO])
		for i = 1, #unigrams do
			local word = unigrams[i]
			if params.word2index[word] == nil then
				tensor[i] = params.word2index[params.UK]
			elseif word == '<zero_rnn>' then
				tensor[i] = 0
			else
				tensor[i] = params.word2index[word]
			end
		end
		if params.label2index ~= nil then
			table.insert(tensors, {tensor, params.label2index[label]})
		else
			table.insert(tensors, {tensor, label2index[label]})
		end
	end
	io.close(fptr)
	return tensors, index2label, label2index
end

-- Function to get train, dev and test tensors
function utils.get_tensors(params)
	print("Getting tensors for training, validating & testing ... ")
	local start = sys.clock()
	params.train_tensors, params.index2label, params.label2index = _get_tensors(params, params.data..'/train.tsv')
	params.val_tensors, _, _ = _get_tensors(params, params.data..'/dev.tsv')
	params.test_tensors, _, _ = _get_tensors(params, params.data..'/test.tsv')
	print('Dataset Size = ('..#params.train_tensors..'/'..#params.val_tensors..'/'..#params.test_tensors..') '..string.format("Done in %.2f minutes.", ((sys.clock() - start)/60)))
end

-- Function to get any layer from nnGraph module
function utils.get_layer(model, name)
	for _, node in ipairs(model.forwardnodes) do
	    if node.data.annotations.name == name then
	        return node.data.module
	    end
	end
	return nil
end

-- Function to initalize word weights
function utils.init_word_weights(params, lookup, file)
	print('initializing the pre-trained embeddings...')
	local start = sys.clock()
	local ic = 0
	for line in io.lines(file) do
		local content = stringx.split(line)
		local word = content[1]
		if params.word2index[word] ~= nil then
			local tensor = torch.Tensor(#content - 1)
			for i = 2, #content do
				tensor[i - 1] = tonumber(content[i])
			end
			lookup.weight[params.word2index[word]] = tensor
			ic = ic + 1
		end
	end
	print(string.format("%d out of %d words initialized.",ic, #params.index2word))
	print(string.format("Done in %.2f seconds.", sys.clock() - start))
end

return utils