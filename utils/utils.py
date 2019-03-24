# -*- coding:utf-8 -*-

import json
import csv
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
from xml.etree.ElementTree import parse
import xml.etree.cElementTree as ET
import glob
import codecs
# from nltk.tokenize import sent_tokenize
import time
import numpy as np


def json_to_csv():
    with open('../data/yelp_academic_dataset_review.json', 'r') as f:
        lines = f.readlines()
        # lines = lines[:int(len(lines)*0.001)]
        new_lines = []
        i = 0
        for line in lines:
            i += 1
            print(str(i), '/', str(len(lines)))
            dic = json.loads(line)
            if dic['stars'] >= 3:
                label = 1
            else:
                label = 0
            review = dic['text'].replace('\t', ' ').replace('\"', '').replace('\n', '')
            if len(review.split(' ')) > 25:
                continue
            review = clean(review)
            if not review:
                continue
            new_lines.append((review, label))
    with open('../data/clean_review.tsv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(new_lines)

def parse_xml():
    tree = ET.ElementTree(file='../data/laptop/Laptops_Train.xml')
    for elem in tree.iter(tag='text'):
        print(elem.text)


def error_sta():
    with open('../train/erros.txt', 'r') as f:
        lines = f.readlines()
        dic = {0:0, 1:0, 2:0}
        for line in lines:
            tmp = str(line).replace('\n', '').split(';')[-1].strip()
            # print (tmp)
            dic[int(tmp)] += 1
        for k, v in dic.items():
            print(k, '---', v)


def prepare_data_for_stanford_parse():
    dirs = ['Train', 'Test_Gold']
    for dir in dirs:
        with open('../data/real_Restaurants_'+dir+'.tsv', 'r') as f:
            lines = f.readlines()
            tmp = []
            for line in lines:
                line = str(line).split('\t')[0]
                tmp.append(line+'\n')
        with open('../data/stanford_parse_data/real_Restaurants_'+dir+'_gcn.txt', 'w') as f:
            for line in tmp:
                f.write(line)


def process_stanford_data():
    dirs = ['real_Restaurants_Train_gcn_res', 'real_Restaurants_Test_Gold_gcn_res']
    for dir in dirs:
        with open('../data/stanford_parse_data/'+dir+'.txt', 'r') as f:
            lines = f.readlines()
            flag = 'tag'
            tags = []; parsers = []
            the_line = ''; tmp = []
            for line in lines:
                if line == '\n' or line == '':
                    if flag == 'tag':
                        tags.append(the_line)
                        flag = 'parser'
                        the_line = ''
                    elif flag == 'parser':
                        parsers.append(tmp)
                        tmp = []
                        flag = 'tag'
                else:
                    if flag == 'tag':
                        the_line = line.replace('\n', '')
                    elif flag == 'parser':
                        tmp.append(line.replace('\n', ''))

        res = []
        for tag, parser in zip(tags, parsers):
            tag = tag.split(' ')
            tokens = []; stanford_pos = []; stanford_head = []; stanford_deprel = []
            for item in tag:
                tokens.append(item.split('/')[0])
                stanford_pos.append(item.split('/')[-1])
            
            for item in parser:
                tmp = item.split('(')
                stanford_deprel.append(tmp[0])
                stanford_head.append(tmp[1].split(',')[0].split('-')[1])
            
            dic = {'tokens':tokens, 'stanford_pos':stanford_pos, 'stanford_head':stanford_head, 'stanford_deprel':stanford_deprel}
            res.append(dic)

        with open('../data/stanford_parse_data/'+dir+'_formal.json', 'w') as f:
            json.dump(res, f)
                

def senti_sta():
    pos = set(); neg = set()
    with open('../data/positive.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            pos.add(line.replace('\n', '').strip())
    with open('../data/negative.txt', 'r') as f:
        for line in lines:
            neg.add(line.replace('\n', '').strip())
    print(len(pos))
    
    def tokenizer(text):
        data = text.split(' ')
        r = []
        for item in data:
            item = item.replace(',','').replace('.', '').replace('?','').replace('!','').replace('-','').replace(';','')
            r.append(item)
        return r
    res = []
    with open('../data/real_Restaurants_Test_Gold.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = str(line).replace('\n', '').split('\t')
            context = set(tokenizer(tmp[0]))
            label = int(tmp[-1])
            if len(context & pos) == 0 and len(context & neg) == 0:
                res.append((tmp[0], tmp[1], label, 0))

    correct = 0
    for item in res:
        print(res)
        if item[-1] == item[-2]:
            correct += 1
    print(len(res))
    print(float(correct)/len(res))
        

def erros_sta():
    dic = {0:[], 1:[], 2:[]}
    with open('../train/erros.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.replace('\n', '').split(';')
            pred = int(tmp[-1].lstrip().rstrip())
            real = int(tmp[-2].lstrip().rstrip())
            dic[pred].append(real)
    print(dic)
    total = len(dic[0]) + len(dic[1]) + len(dic[2])
    print(len(dic[0])/total)
    print(len(dic[1])/total)
    print(len(dic[2])/total)
    


def get_high_pre():
    with open('../train/acc_records.txt', 'r') as f:
        lines = f.readlines()
        accs = []
        for line in lines:
            line = str(line).replace('\n', '').split(' ')
            line = list(map(float, line))
            accs.append(max(line))
        print('max: ', max(accs))
        print('min: ', min(accs))
        print('mean: ', np.mean(accs))

if __name__ == '__main__':
    # prepare_data_for_stanford_parse()
    # process_stanford_data()
    # senti_sta()
    # erros_sta()
    get_high_pre()
