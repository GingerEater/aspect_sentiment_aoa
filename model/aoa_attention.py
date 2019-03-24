import sys
from os import path
import torch.nn as nn
import torch
import torch.nn.functional as F


class AOA_Attention(nn.Module):
    def __init__(self, gpu=True):
        super(AOA_Attention, self).__init__()
        self.use_gpu = gpu

    def sequence_mask(self, input_weight, seq_lengths, target_lengths):
        input_weight_bk = input_weight

        def get_softmax_att(tmp_input_weight, seq_len, batch_size):
            input_weight = tmp_input_weight.contiguous()
            mask_ = torch.arange(1, seq_len+1).type(torch.LongTensor).repeat(batch_size, 1).cuda()  # batch, seq_len
            mask_ = mask_.le(seq_lengths.unsqueeze(1)).unsqueeze(1)
            input_weight.data.masked_fill_(1-mask_, -10000)
            align_weight = F.softmax((input_weight.view(-1,seq_len)), dim=1).view(batch_size,-1,seq_len)
            return align_weight
        
        batch_size = input_weight.size(0)
        
        seq_len  = input_weight.size(2)
        align_weight_target_seq = get_softmax_att(input_weight, seq_len, batch_size)  # batch, target_len, seq_len

        if not torch.equal(input_weight_bk, input_weight):
            print('wrong!!!')
        target_len = input_weight.size(1)
        align_weight_seq_target = get_softmax_att(torch.transpose(input_weight, 1, 2), target_len, batch_size)  # batch, seq_len, target_len
        
        return align_weight_target_seq, align_weight_seq_target


    def get_final_repsentation(self, seq_out, seq_length, align_weight_target_seq, align_weight_seq_target):
        align_weight = torch.sum(align_weight_seq_target, dim=1)
        align_weight = align_weight / seq_length.unsqueeze(1).type(torch.cuda.FloatTensor)  # pad的不参与计算均值
        align_weight = align_weight.unsqueeze(1)  # batch, 1, target_len
        # align_weight = align_weight_seq_target
        # align_weight = torch.mean(align_weight, 1, True)

        final_att = torch.bmm(torch.transpose(align_weight_target_seq, 1, 2), torch.transpose(align_weight, 1, 2))  # batch, seq_len, 1

        final_repsentation = torch.bmm(torch.transpose(seq_out, 1, 2), final_att)  # batch, hidden, 1

        batch_size = final_repsentation.size(0); hidden = final_repsentation.size(1)
        final_repsentation = final_repsentation.view(batch_size, hidden)  # batch, hidden

        return final_repsentation, final_att

    # targe_vec: batch,target_len,hidden
    # seq_out:   batch,seq_len,hidden
    def forward(self, seq_out, seq_length, target_vec, target_length):
        input_weight = torch.bmm(target_vec, seq_out.transpose(1, 2)).contiguous()  # batch, target_len, seq_len
        
        align_weight_target_seq,  align_weight_seq_target = self.sequence_mask(input_weight, seq_length, target_length)

        final_repsentation, final_att = self.get_final_repsentation(seq_out, seq_length, align_weight_target_seq,  align_weight_seq_target)

        return  final_repsentation, align_weight_target_seq
