import os
import sys
sys.path.append('./')
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils.arguments_parse import args
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re
from data_preprocessing import tools,build_word_vocab
import jieba
from tqdm import tqdm

tokenizer=tools.get_tokenizer()
word2id, _ = tools.load_vocab()


def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        lines=f.readlines()
        for l in lines:
            l = json.loads(l)
            arguments={}
            for arg in l['entity_list']:
                arguments[arg['type']]=arg['argument']
            if arguments != {}:
                D.append((l['text'], arguments))
    return D


def cut_word(sentence):

    word_list=list(jieba.cut(sentence,cut_all=True))

    token_list = ['[CLS]']+tokenizer.tokenize(sentence)
    token_list = [token.replace('##','') for token in token_list]
    win=['。' for i in range(6)]
    token_list = win + token_list + win

    char_word=[]
    for i,token in enumerate(token_list[6:-6]):
        char_word_list =[]
        sub_text=''.join(token_list[i:i+12])
        for word in word_list:
            if token in word and word in sub_text:
                char_word_list.append(word)
        char_word_id=[0,0,0,0]
        for word in char_word_list:
            if token == word:
                char_word_id[3]=word2id[word]
            else:
                index=word.index(token)
                if index==0:
                    char_word_id[0]=word2id[word]
                elif index==len(word)-1:
                    char_word_id[2]=word2id[word]
                else:
                    char_word_id[1]=word2id[word]
        char_word.append(char_word_id)

    while len(char_word)<args.max_length:
        char_word.append([0 for i in range(4)])
    return char_word[:args.max_length] 


def encoder(sentence,argument):
    label2id,id2label,num_labels = tools.load_schema()
    encode_dict=tokenizer.encode_plus(sentence,max_length=args.max_length,pad_to_max_length=True)
    encode_sent=encode_dict['input_ids']
    token_type_ids=encode_dict['token_type_ids']
    attention_mask=encode_dict['attention_mask']
    label=[0 for i in range(args.max_length)]
    for key,value in argument.items():
        encode_arg=tokenizer.encode(value)
        start_idx=tools.search(encode_arg[1:-1],encode_sent)
        label[start_idx]= label2id[key] * 2 + 1
        for i in range(1, len(encode_arg[1:-1])):
            label[start_idx + i] = label2id[key] * 2 + 2
    return encode_sent,token_type_ids,attention_mask,label


def data_pre(file_path):
    data=load_data(file_path)

    result=[]
    for (text, arguments) in tqdm(data):
        encode_sent,token_type_ids,attention_mask,label=encoder(text,arguments)
        token_word = cut_word(text)
        tmp = {}
        tmp['input_ids'] = encode_sent
        tmp['input_seg'] = token_type_ids
        tmp['input_mask'] = attention_mask
        tmp['labels'] = label
        tmp['token_word']=token_word
        result.append(tmp)
    return result


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "input_seg": torch.tensor(item['input_seg']).long(),
            "input_mask": torch.tensor(item['input_mask']).float(),
            "labels": torch.tensor(item['labels']).long(),
            "token_word":torch.tensor(item['token_word']).long()
        }
        return one_data


def yield_data(file_path):
    tmp = MyDataset(data_pre(file_path))
    return DataLoader(tmp, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':

    data = data_pre(args.train_path)

    # print(input_ids_list[0])
    # print(token_type_ids_list[0])
    # print(start_labels[0])
