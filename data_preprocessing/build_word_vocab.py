import json
import jieba
import sys
sys.path.append('./')
from utils.arguments_parse import args
from data_preprocessing import tools
from tqdm import tqdm

jieba.add_word('[CLS]')
jieba.add_word('[SEP]')
jieba.add_word('[unused1]')


def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        lines=f.readlines()
        for l in lines:
            l = json.loads(l)
            D.append(l['text'])
    return D


def save_vocab():

    word_list=[]
    sentences = load_data(args.train_path)
    sentences.extend(load_data(args.test_path))
    for sent in tqdm(sentences):
        tmp_word_list=list(jieba.cut(sent,cut_all=True))
        for word in tmp_word_list:
            if word not in word_list:
                word_list.append(word)
                
    vocab_lenth=len(word_list)
    word2id={}
    id2word={}
    for i,word in enumerate(word_list):
        word2id[word]=i
        id2word[i]=word

    with open('./data/vocab.json','w',encoding='utf8') as f:
        tmp=json.dumps(word2id,ensure_ascii=False)
        f.write(tmp)

    return word2id,id2word,vocab_lenth


def load_vocab():
    with open('./data/vocab.json','r',encoding='utf8') as f:
        lines=f.readlines() 
        for line in lines:
            word2id=json.loads(line)

    return word2id,len(word2id)

if __name__=="__main__":
    save_vocab()
    # word,l=load_vocab()
