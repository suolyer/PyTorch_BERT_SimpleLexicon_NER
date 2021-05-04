import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel,BertConfig
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
from data_preprocessing import tools
from data_preprocessing import build_word_vocab
from .loss_function import crf

tokenizer=tools.get_tokenizer()
_,vocab_size=build_word_vocab.load_vocab()
num_labels=tools.get_labels_num()


class myModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.bert_config = BertConfig.from_pretrained(pre_train_dir)
        self.roberta_encoder.resize_token_embeddings(len(tokenizer))
        self.encoder_linear = torch.nn.Sequential(torch.nn.Linear(in_features=self.bert_config.hidden_size, 
                                                                    out_features=self.bert_config.hidden_size),
                                                  torch.nn.Tanh(),
                                                  torch.nn.Dropout(p=dropout_rate))
        
        self.lstm=torch.nn.LSTM(input_size=2*self.bert_config.hidden_size,
                                hidden_size=2*self.bert_config.hidden_size,
                                num_layers=1,
                                batch_first=True,
                                dropout=dropout_rate,
                                bidirectional=True)

        self.word_embeddings = nn.Embedding(vocab_size, self.bert_config.hidden_size)
        self.word_layer1 = torch.nn.Sequential(torch.nn.Linear(in_features=self.bert_config.hidden_size, out_features=1024),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=dropout_rate))
        self.word_layer2 = torch.nn.Linear(in_features=1024, out_features=self.bert_config.hidden_size)

        self.attention_layer1 = torch.nn.Sequential(torch.nn.Linear(in_features=1024, out_features=512),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Dropout(p=dropout_rate))
        self.attention_layer2 = torch.nn.Linear(in_features=512, out_features=1)

        self.crf = crf.CRF(num_tags=num_labels, batch_first=True)
        self.classifier = torch.nn.Linear(in_features=4*768, out_features=num_labels)



    def forward(self, input_ids, input_mask, input_seg, token_word, labels=None):
        bert_output = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)
        encoder_rep = bert_output[0]

        batch_size,seq_len,hidden=encoder_rep.shape
        token_word = self.word_embeddings(token_word)
        token_word = self.word_layer1(token_word)
        word_emb = self.word_layer2(token_word)

        attention = self.attention_layer1(token_word)
        attention = self.attention_layer2(attention)
        attention = attention.view(batch_size,seq_len,4)
        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = attention.unsqueeze(-1)
        attention = attention.repeat(1,1,1,768)

        word_emb = torch.mul(word_emb,attention)
        word_emb = torch.sum(word_emb,dim=2).view(-1,seq_len,hidden)
        encoder_rep = torch.cat([encoder_rep,word_emb],dim=-1)

        sequence_output,_ = self.lstm(encoder_rep)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions=logits, tags=labels, mask=input_mask)
            outputs = self.crf.decode(logits, input_mask)
            return loss, outputs
        else:
            tags = self.crf.decode(logits, input_mask)
            outputs = tags
            return outputs