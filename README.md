## Tweet Classification using RNN and CNN

[Torch](http://torch.ch) implementation of a tweet classifier with GPUs. It allows training using a deep CNN and RNN models. We also provide option to choose different activation functions of RNNs such as GRU, LSTM and their bi-directional variants.

The CNN model is from
[Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181),
Kim et al. EMNLP 2014.

This project is maintained by [Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/). You are welcome to post any issues on the issues page.

### Quickstart

We will be working with a sample data from [Sentiment140](http://help.sentiment140.com/for-students/). The data has 6000, 600 and 359 training, validation and test tweets respectively along with their labels in a tab-separated format.

Run all the models at one shot

```
bash run_all.sh
```

This will run CNN, RNN, Bi-RNN, GRU, Bi-GRU, LSTM and Bi-LSTM serially in a GPU. The default settings for CNNs are taken from the original paper while RNNs settings are leveraged from [TreeLSTM](https://github.com/stanfordnlp/treelstm).


### Training options

#### Convolutional Neural Network (`cnn.lua`)

* `num_feat_maps`: number of feature maps after 1st convolution
* `kernels`: kernel sizes of convolutions, table format. (ex: 3,4,5)
* `dropout_p`: dropout for regularization
* `L2s`: L2 normalize weights
* `optim_method`: which gradient descent algorithm to choose? adadelta or adam 

#### Recurrent Neural Network (`*rnn.lua`)

* `layers`: size of each hidden layer in RNN (ex: for 3-layer net, give it as {150,150,150})
* `rnn_type`: which variant of RNN to use? RNN or GRU or LSTM
* `seq_length`: number of timessteps to unroll for
* `optim_method`: which gradient descent algorithm to choose? adadelta or adam or adagrad
* `learning_rate`: learning rate to be given to the optmizer.

#### Miscellaneous `*.lua`

* `data`: path to the data.
* `seed`: change the random seed for random numbers in torch - use that option to train alternate models for ensemble
* `glove_dir`: directory for accesssing the pre-trained twitter-specific glove word embeddings. You can download them [here](http://nlp.stanford.edu/projects/glove/)
* `pre_train`: should we initialize word embeddings with pre-trained vectors?
* `dim`: dimensionality of word embeddings.
* `min_freq`: words that occur less than <int> times will not be taken for training. They are substituted with special token.
* `num_epochs`: number of full passes through the training data
* `bsize`: mini-Batch size


#### Torch Dependencies
* nn
* optim
* xlua
* cunn
* cutorch
* cudnn
* nngraph
* dpnn
* rnn

#### Licence
MIT
