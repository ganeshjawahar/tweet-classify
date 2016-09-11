require('nngraph')
require('cudnn')

function get_model(config)
	local inputs = {nn.Identity()()}
	local lookup = nn.LookupTable(config.ngram_lookup_rows, config.ngram_lookup_cols)(inputs[1]):annotate{name = 'ngram_lookup'}
	local kernels = stringx.split(config.kernels, ',')
	local layer1 = {}
	for i = 1, #kernels do
		local conv
		local conv_layer
		local max_time
		conv = cudnn.SpatialConvolution(1, config.num_feat_maps, config.ngram_lookup_cols, tonumber(kernels[i]))
		conv_layer = nn.Reshape(config.num_feat_maps, config.max_words - kernels[i] + 1, true)(
			conv(nn.Reshape(1, config.max_words, config.ngram_lookup_cols, true)(lookup)))
		max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
		conv.weight:uniform(-0.01, 0.01)
		conv.bias:zero()
		conv.name = 'convolution'
		table.insert(layer1, max_time)
	end
	local conv_layer_concat
	if #layer1 > 1 then
		conv_layer_concat = nn.JoinTable(2)(layer1)
	else
		conv_layer_concat = layer1[1]
	end
	local linear = nn.Linear((#layer1) * config.num_feat_maps, config.num_classes)
	linear.weight:normal():mul(0.01)
	linear.bias:zero()
	local output = linear(nn.Dropout(config.dropout_p)(conv_layer_concat)):annotate{name = 'linear_pred_layer'}
	model = nn.gModule(inputs, {output})
	return model
end