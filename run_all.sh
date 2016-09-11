echo 'running CNN...'
th cnn.lua
echo 'running RNN...'
th rnn.lua -rnn_type rnn
echo 'running Bi-Directional RNN...'
th bi-rnn.lua -rnn_type rnn
echo 'running GRU...'
th rnn.lua -rnn_type gru
echo 'running Bi-Directional GRU...'
th bi-rnn.lua -rnn_type gru
echo 'running LSTM...'
th rnn.lua -rnn_type lstm
echo 'running Bi-Directional LSTM...'
th bi-rnn.lua -rnn_type lstm