# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import sys
sys.path.append('..')
import config.config_cla as args
import pickle
from model.cla import Cla
import numpy as np
from embeddings import GloveEmbedding
from sklearn.metrics import classification_report
import spacy
import time
# SEED = 0
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# np.random.seed(SEED)

spacy_en = spacy.load('en_core_web_sm')
def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]
srcF = Field(sequential=True, batch_first=True, tokenize=tokenizer, lower=True, include_lengths=True)
tgtF = Field(sequential=True, batch_first=True, tokenize=tokenizer, lower=True, include_lengths=True)
labelF = Field(sequential=False, batch_first=True, use_vocab=False)


print('load data...')
all_data = TabularDataset(path=args.all_data, format='tsv', fields=[('src', srcF), ('tgt', tgtF), ('label', labelF)])
train = TabularDataset(path=args.train_data, format='tsv', fields=[('src', srcF), ('tgt', tgtF), ('label', labelF)])
dev = TabularDataset(path=args.dev_data, format='tsv', fields=[('src', srcF), ('tgt', tgtF), ('label', labelF)])


srcF.build_vocab(all_data, min_freq=1)
vocab = srcF.vocab
tgtF.vocab = vocab
args.vocab_size = len(vocab)

g = GloveEmbedding('common_crawl_840', d_emb=300)
embedding = []
for i in range(len(vocab)):
    if not g.emb(vocab.itos[i])[0]:
        embedding.append(np.random.uniform(-0.01, 0.01, size=(1, 300))[0])
    else:
        embedding.append(np.array(g.emb(vocab.itos[i])))
embedding = np.array(embedding, dtype=np.float32)
args.pre_embedding = True
args.embedding = embedding
args.update_embedding = False

print('build batch iterator...')
train_batch_iterator = BucketIterator(
    dataset=train, batch_size=args.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    repeat=False
)
dev_batch_iteraor = BucketIterator(
    dataset=dev, batch_size=args.batch_size,
    sort=False, sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    repeat=False
)

cla = Cla(args)

weight_p, bias_p = [],[]
for name, p in cla.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay':1e-4},
                              {'params': bias_p, 'weight_decay':0}], lr=args.lr)
# optimizer = torch.optim.Adam(cla.parameters(), lr=args.lr)
loss_func = nn.CrossEntropyLoss()
cla.cuda()
loss_func.cuda()

print('begin training...')
best_acc = float('-inf')

acc_on_dev = []

for epoch in range(1, args.epochs+1):

    print('第', str(epoch), '轮训练')
    batch_generator = train_batch_iterator.__iter__()

    total_loss = []
    correct = 0
    total = 0

    y_true, y_pred = [], []
    target_names = ['neutral', 'negative', 'positive']

    train_error = []
    cla.train()
    last_loss_data = float('inf')
    loss_drop_counter = 0
    for batch in batch_generator:
        srcs, src_lengths = getattr(batch, 'src')
        tgts, tgt_lengths = getattr(batch, 'tgt')
        labels = getattr(batch, 'label')
        
        pred, _ = cla(srcs.cuda(), src_lengths.cuda(), tgts.cuda(), tgt_lengths.cuda() )
        loss = loss_func(pred.cuda(), labels.cuda() )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # cal loss and acc on train data
        pred = pred.max(1)[1]
        y_true.extend(labels.data.numpy().tolist())
        y_pred.extend(pred.cpu().data.numpy().tolist())
        total_loss.append(loss.item())
        correct += pred.cpu().long().eq(labels.data.long()).sum()
        total += len(srcs)
        # if len(total_loss) == 100:
        #     print('Train')
        #     print('loss: ', str(round(np.mean(total_loss), 2)), 'acc: ', str(round(100.*correct.data.numpy()/total, 2)))
        #     correct = 0
        #     total_loss = []
        #     total = 0

        loss_data = loss.item()
        # loss_data = np.mean(total_loss)
        if loss_data >= last_loss_data:
            loss_drop_counter += 1
            if loss_drop_counter >= 3:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    loss_drop_counter = 0
        else:
            loss_drop_counter = 0
        last_loss_data = loss.item()
        # last_loss_data = loss_data


    # print('准确率', str(round(100.*correct.data.numpy()/total, 2)) )
    # print('在训练集上的分类报告')
    # print(classification_report(y_true, y_pred, target_names=target_names))

    # cal loss and acc on dev data
    batch_generator = dev_batch_iteraor.__iter__()
    correct = 0
    total = 0

    y_true, y_pred = [], []
    target_names = ['neutral', 'negative', 'positive']

    error = []
    right = []
    total_list = []
    cla.eval()
    for batch in batch_generator:
        srcs, src_lengths = getattr(batch, 'src')
        tgts, tgt_lengths = getattr(batch, 'tgt')
        labels = getattr(batch, 'label')
        pred, final_att = cla(srcs.cuda(), src_lengths.cuda(), tgts.cuda(), tgt_lengths.cuda() )
        pred = pred.max(1)[1]
        y_true.extend(labels.data.numpy().tolist())
        y_pred.extend(pred.cpu().data.numpy().tolist())
        correct += pred.cpu().long().eq(labels.data.long()).sum()
        total += len(srcs)

        
        for i in range(len(srcs)):
            if pred[i].cpu() == labels[i]:
                tmp_list = []
                for k in range(tgt_lengths[i]):
                    src = ''
                    for j in range(src_lengths[i]):
                        src += srcF.vocab.itos[srcs[i][j]]+'('+str(round(final_att[i][k][j].item(), 3) )+')' + ' '
                    src += ' ; '+tgtF.vocab.itos[tgts[i][k]] + ' ; '+str(pred[i].cpu().item())+' ; ' + str(labels[i].item()) + '\n'
                    tmp_list.append(src)
                total_list.append(tmp_list)
        
        # for i in range(len(srcs)):
        #     if pred[i].cpu() == labels[i]:
        #         src = ''
        #         for j in range(src_lengths[i]):
        #             src += srcF.vocab.itos[srcs[i][j]]+'('+str(round(final_att[i][j].item(), 3) )+')' + ' '
        #         tgt = ''
        #         for j in range(tgt_lengths[i]):
        #             tgt += tgtF.vocab.itos[tgts[i][j]] + ' '
        #         error.append(src+'; '+tgt+'; '+str(pred[i].cpu().item())+' ; '+str(labels[i].item())+'\n')

    # with open('error.txt', 'w') as f:
    #     for line in error:
    #         f.write(line)
    with open('right.txt', 'w') as f:
        for tmp_list in total_list:
            for line in tmp_list:
                f.write(line)
            f.write('\n')
                
    acc = round(100.*correct.data.numpy()/total, 2)
    acc_on_dev.append(str(acc))
    print('Dev')
    print('acc: ', str(acc))

    # print('在验证集上的分类报告')
    # print(classification_report(y_true, y_pred, target_names=target_names))

    # save model
    if acc > best_acc:
        torch.save(cla.state_dict(), args.model_path)
        best_acc = acc
    
with open('acc_records.txt', 'a') as f:
    f.write(' '.join(acc_on_dev)+'\n')
