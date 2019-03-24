# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.aoa_attention import AOA_Attention
import numpy as np


class Cla(nn.Module):
    def __init__(self, args):
        super(Cla, self).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.bidirectional = args.bidirectional
        self.vocab_size = args.vocab_size
        self.input_dropout = nn.Dropout(args.input_dropout)
        self.output_dropout = args.output_dropout
        self.cato_num = args.cato_num
        self.layer_num = args.layer_num

        if args.rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif args.rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: {0}'.format(args.rnn_cell))

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=args.pad_id)
        if args.pre_embedding:
            self.embedding.weight = nn.Parameter(torch.from_numpy(args.embedding))
        self.embedding.requires_grad = args.update_embedding

        self.rnn1_src = self.rnn_cell(self.embedding_size, self.hidden_size, self.layer_num, batch_first=True,
                                 bidirectional=self.bidirectional)
        self.rnn1_tgt = self.rnn_cell(self.embedding_size, self.hidden_size, self.layer_num, batch_first=True,
                                 bidirectional=self.bidirectional)
        self.initial('lstm')

        self.aoa_attention = AOA_Attention()

        self.out = nn.Linear(self.hidden_size*2, self.cato_num)
        self.initial('linear')

    def initial(self, cato):
        if cato == 'lstm':
            torch.nn.init.uniform_(self.rnn1_src.weight_ih_l0, a=-0.0001, b=0.0001)
            torch.nn.init.uniform_(self.rnn1_src.weight_hh_l0, a=-0.0001, b=0.0001)
            torch.nn.init.uniform_(self.rnn1_tgt.weight_ih_l0, a=-0.0001, b=0.0001)
            torch.nn.init.uniform_(self.rnn1_tgt.weight_hh_l0, a=-0.0001, b=0.0001)
            torch.nn.init.constant_(self.rnn1_src.bias_ih_l0, 0.0)
            torch.nn.init.constant_(self.rnn1_src.bias_hh_l0, 0.0)
            torch.nn.init.constant_(self.rnn1_tgt.bias_ih_l0, 0.0)
            torch.nn.init.constant_(self.rnn1_tgt.bias_hh_l0, 0.0)
        if cato == 'linear':
            torch.nn.init.uniform_(self.out.weight, a=-0.0001, b=0.0001)
            torch.nn.init.constant_(self.out.bias, 0.0)
    
    def lstm_process(self, inputs, input_lengths, model, kind):
        if kind == 'tgt':
            inputs = inputs.data.cpu().numpy()
            input_lengths = input_lengths.data.cpu().numpy()

            sort_idx = np.argsort(-input_lengths)
            inputs = inputs[sort_idx]
            input_lengths = input_lengths[sort_idx]
            unsort_idx = np.argsort(sort_idx)

            inputs = torch.from_numpy(inputs)
            input_lengths = torch.from_numpy(input_lengths)

            inputs = nn.utils.rnn.pack_padded_sequence(inputs.cuda(), input_lengths.cuda(), batch_first=True)
            output, _ = model(inputs)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            output = output[unsort_idx]
            input_lengths = input_lengths[unsort_idx]

        elif kind == 'src':
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
            output, _ = model(inputs)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, input_lengths

    def forward(self, src, src_length, tgt, tgt_length, initial_h=None, initial_c=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # 过dropout
        src = self.input_dropout(src)
        tgt = self.input_dropout(tgt)

        # 第一个lstm
        src_output, src_length = self.lstm_process(src, src_length, self.rnn1_src, 'src')
        tgt_output, tgt_length = self.lstm_process(tgt, tgt_length, self.rnn1_tgt, 'tgt')

        # aoa
        final_repsentation, final_att = self.aoa_attention(src_output, src_length, tgt_output, tgt_length)
        # final_att = final_att.view(final_att.size(0), final_att.size(1))
        # target_final_repsentation = self.aoa_attention(tgt_output, tgt_length, src_output, src_length)

        # r = torch.cat([final_repsentation, target_final_repsentation], dim=1)

        out = self.out(final_repsentation)

        # out = F.tanh(out)

        return out, final_att
