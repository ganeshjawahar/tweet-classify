## Tweet Classification using RNN and CNN

[Torch](http://torch.ch) implementation of a tweet classifier with GPUs. It allows training using a deep CNN and RNN models. We also provide option to choose different RNNs such as GRU, LSTM and their bi-directional variants.

The CNN model is from
[Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181),
Kim et al. EMNLP 2014.

This project is maintained by [Ganesh J](https://researchweb.iiit.ac.in/~ganesh.j/). You are welcome to post any issues on the issues page.

### Quickstart

We will be working with sample data from [Sentiment140](http://help.sentiment140.com/for-students/). The data has 6000, 600 and 359 training, validation and test tweets respectively along with their labels in tab-separated format.

Run all the models at one shot

```
bash run_all.sh
```

This will run CNN, RNN, Bi-RNN, GRU, Bi-GRU, LSTM and Bi-LSTM serially in a GPU. The default settings for CNNs are taken from the original paper while for RNNs they are taken from [TreeLSTM](https://github.com/stanfordnlp/treelstm).


### Training options

#### Convolutional Neural Network (`cnn.lua`)

* `num_feat_maps`: Number of feature maps after 1st convolution
* `kernels`: Kernel sizes of convolutions, table format.
* `dropout_p`: p for dropout
* `L2s`: L2 normalize weights
* `optim_method`: Which gradient descent algorithm to choose? adadelta or adam 

#### Recurrent Neural Network (`*rnn.lua`)

* `layers`: Size of each hidden layer in RNN (ex: for 3-layer net, give it as {150,150,150})
* `rnn_type`: Which variant of RNN to use? RNN or GRU or LSTM
* `seq_length`: Number of timessteps to unroll for
* `optim_method`: Which gradient descent algorithm to choose? adadelta or adam or adagrad
* `learning_rate`: Learning rate to be given to the optmizer.

#### Miscellaneous `*.lua`

* `data`: Path to the data.
* `seed`: Change the random seed for random numbers in torch - use that option to train alternate models for ensemble
* `glove_dir`: Directory for accesssing the pre-trained twitter-specific glove word embeddings. You can download them [here](http://nlp.stanford.edu/projects/glove/)
* `pre_train`: Should we initialize word embeddings with pre-trained vectors?
* `dim`: Dimensionality of word embeddings.
* `min_freq`: Words that occur less than <int> times will not be taken for training. They are substituted with special token.
* `num_epochs`: Number of full passes through the training data
* `bsize`: Mini-Batch size


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