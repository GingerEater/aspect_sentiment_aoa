import sys
from os import path
import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size=150, gpu=True):
        super(Attention, self).__init__()
        self.use_gpu = gpu

        self.fc_atten = nn.Linear(hidden_size,hidden_size)


    def sequence_mask(self, input_weight, input_lengths, is_self=False):
        batch_size = input_weight.size(0)

        seq_len  = input_weight.size(2)

        mask_ = torch.arange(1, seq_len+1).type(torch.LongTensor).repeat(batch_size,1)  # bach, seq_len
        if self.use_gpu:
            mask_ = mask_.cuda()
        
        if is_self:  # self-attention，需要将自己也mask掉
            mask_ = mask_.le(input_lengths.unsqueeze(1))  # mask_: batch, seq_len，那些pad的地方会被置为0
            lengths = torch.arange(0, batch_size).unsqueeze(1)
            mask_ = mask.scatter_(1, lengths, 0).unsqueeze(1)  # mask_: batch, 1, seq_len，每一个句子，其自身会被置为0，这样attention时就会忽略自身
        else:
            mask_ = mask_.le(input_lengths.unsqueeze(1)).unsqueeze(1)

        input_weight.data.masked_fill_(1-mask_, -10000)

        # align_weight: batch,-1,seq_len
        align_weight = F.softmax((input_weight.view(-1,seq_len)), dim=1).view(batch_size,-1,seq_len)

        return align_weight

    # targe_vec: batch,len1,hidden
    # seq_out:   batch,len2,hidden
    def forward(self, seq_out, seq_length, target_vec, is_self=False):
        input_weight = torch.tanh(torch.bmm(target_vec, seq_out.transpose(1, 2))).contiguous()  # batch, seq_len1, seq_len2
        align_weight = self.sequence_mask(input_weight, seq_length)  # batch, seq_len1, seq_len2
        
        return  align_weight
