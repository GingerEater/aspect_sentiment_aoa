# -*- coding:utf-8 -*-

embedding_size = 300
vocab_size = 0
cato_num = 3
sos_id = 0
eos_id = 0
pad_id = 0
input_dropout = 0.2
output_dropout = 0.2
hidden_size = 150
layer_num = 1
pre_embedding = False
embedding = None
update_embedding = True
rnn_cell = 'lstm'
bidirectional = True
lr = 0.001
batch_size = 25
epochs = 30
all_data = '../data/real_Restaurants_All.tsv'
train_data = '../data/real_Restaurants_Train.tsv'
dev_data = '../data/real_Restaurants_Test_Gold.tsv'
model_path = '..//gen/cla.pkl'
embedding_path = '../gen/embedding_mat.pkl'
vocab_path = '../gen/vocab.pkl'
