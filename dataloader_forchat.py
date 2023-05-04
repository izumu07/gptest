import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from torch.utils.data import DataLoader
import time

class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        
        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('iemocap/iemocap_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

        print(len(self.keys), self.keys)
    def __getitem__(self, index):
        vid = self.keys[index]
        print(vid)
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               self.sentences[vid],\
               self.sentences[vid]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]
    

class MELDRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping = {'sadness':3, 'joy': 4, 'neutral':0, 'surprise':1, 'anger':6, 'disgust':5, 'fear':2}
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('meld/meld_features_comet.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

        print(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        print(vid)
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor(self.speakers[vid]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               self.sentences[vid],\
               self.sentences[vid]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]

class DailyDialogueRobertaCometDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'sadness':3, 'joy': 4, 'neutral':0, 'surprise':1, 'anger':6, 'disgust':5, 'fear':2}
        '''
        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('dailydialog/dailydialog_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        print(vid)
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               self.sentences[vid],\
               self.sentences[vid]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]

class EmoryNLPRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''
        
        self.speakers, self.emotion_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainId, self.testId, self.validId \
        = pickle.load(open('emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')
        
        sentiment_labels = {}
        for item in self.emotion_labels:
            array = []
            # 0 negative, 1 neutral, 2 positive
            for e in self.emotion_labels[item]:
                if e in [1, 4, 6]:
                    array.append(0)
                elif e == 3:
                    array.append(1)
                elif e in [0, 2, 5]:
                    array.append(2)
            sentiment_labels[item] = array

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('emorynlp/emorynlp_features_comet.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]
            
        if classify == 'emotion':
            self.labels = self.emotion_labels
        elif classify == 'sentiment':
            self.labels = sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]), \
               self.sentences[vid], \
               self.sentences[vid]

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else dat[i].tolist() for i in dat]

def get_IEMOCAP_loaders(batch_size=32, num_workers=0, dataset = "fuck", pin_memory=False):
    if dataset == "DD":
        trainset = DailyDialogueRobertaCometDataset('train')
        validset = DailyDialogueRobertaCometDataset('valid')
        testset = DailyDialogueRobertaCometDataset('test')
    else:
        if dataset == "MELD3":
            trainset = MELDRobertaCometDataset('train', classify='sentiment')
            validset = MELDRobertaCometDataset('valid', classify='sentiment')
            testset = MELDRobertaCometDataset('test', classify='sentiment')
        else:
            trainset = MELDRobertaCometDataset('train')
            validset = MELDRobertaCometDataset('valid')
            testset = MELDRobertaCometDataset('test')
        # if dataset == "EmoryNLP3":
        #     trainset = EmoryNLPRobertaCometDataset('train', classify='sentiment')
        #     validset = EmoryNLPRobertaCometDataset('valid', classify='sentiment')
        #     testset = EmoryNLPRobertaCometDataset('test', classify='sentiment')
        # else:
        #     trainset = EmoryNLPRobertaCometDataset('train')
        #     validset = EmoryNLPRobertaCometDataset('valid')
        #     testset = EmoryNLPRobertaCometDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

import numpy as np

def cut_list(lists, cut_len):
    """
    将列表拆分为指定长度的多个列表
    :param lists: 初始列表
    :param cut_len: 每个列表的长度
    :return: 一个二维数组 [[x,x],[x,x]]
    """
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)

    return res_data

import openai
import re

openai.api_key = "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"
from sklearn.metrics import f1_score, accuracy_score
if __name__ == '__main__':

    dataset = "MELD"

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(batch_size=1,
                                                                  num_workers=0, dataset=dataset)
    ######################################################################
    if dataset == 'DD':
        # IEMOCAP
        # str_label = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated', 'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated', 'Unknown', 'unknown']
        # mapping = {'Happy': 0, 'Sad': 1, 'Neutral': 2, 'Angry': 3, 'Excited': 4, 'Frustrated': 5, 'happy': 0, 'sad': 1, 'neutral': 2, 'angry': 3, 'excited': 4, 'frustrated': 5, 'Unknown': 2, 'unknown': 2}

        # DD
        str_label = ['happiness', 'neutral', 'fear', 'disgust', 'anger', 'surprise', 'sadness', 'Happiness', 'Neutral', 'Fear', 'Disgust', 'Anger', 'Surprise', 'Sadness']
        mapping = {'happiness': 0, 'neutral': 1, 'fear': 4, 'disgust': 6, 'anger': 2, 'surprise': 5, 'sadness': 3, 'Happiness': 0, 'Neutral': 1, 'Fear': 4, 'Disgust': 6, 'Anger': 2, 'Surprise': 5, 'Sadness': 3}
    else:
        # MELD
        if dataset == 'MELD3':
            str_label = {'Positive', 'Negative', 'Neutral', 'positive', 'negative', 'neutral'}
            mapping = {'Positive': 1, 'Negative': 2, 'Neutral': 0, 'positive': 1, 'negative': 2, 'neutral': 0}
        else:
            str_label = {'sadness', 'joy', 'neutral', 'surprise', 'anger', 'disgust', 'fear', 'Sadness', 'Joy', 'Neutral', 'Surprise', 'Anger', 'Disgust', 'Fear'}
            mapping = {'sadness': 3, 'joy': 4, 'neutral': 0, 'surprise': 1, 'anger': 6, 'disgust': 5, 'fear': 2, 'Sadness': 3, 'Joy': 4, 'Neutral': 0, 'Surprise': 1, 'Anger': 6, 'Disgust': 5, 'Fear': 2}

        # EmoryNLP
        # if dataset == 'EmoryNLP3':
        #     str_label = {'Positive', 'Negative', 'Neutral', 'positive', 'negative', 'neutral'}
        #     mapping = {'Positive': 2, 'Negative': 0, 'Neutral': 1, 'positive': 2, 'negative': 0, 'neutral': 1}
        # else:
        #     str_label = {'Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared', 'joyful', 'mad', 'peaceful', 'neutral', 'sad', 'powerful', 'scared'}
        #     mapping = {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6, 'joyful': 0, 'mad': 1, 'peaceful': 2, 'neutral': 3, 'sad': 4, 'powerful': 5, 'scared': 6}

    ######################################################################
    trs = []
    trs_number = []
    labels = []
    masks = []

    for data in test_loader:
        r1, r2, r3, r4, \
        xIntent, xAttr, xNeed, xWant, xEffect, xReact, \
        oWant, oEffect, oReact, \
        qmask, umask, label, uttrs = [d for d in data[:-1]]
        index = 1
        messages = []
        masks += umask.numpy().tolist()[0]
        labels += label.numpy().tolist()[0]
        ones = {}
        ones["role"] = "system"
        ones["content"] = "Here is a new dialogue with " + str(len(label.numpy().tolist()[0])) + " utterances"
        messages.append(ones)
        for u, q in zip(uttrs[0], qmask):
            ones = {}
            utterance = str(index) + '. ' + 'Speaker' + str(np.argmax(q[0]).item()) + ': ' + u
            index += 1
            ones["role"] = "user"
            ones["content"] = utterance
            messages.append(ones)
        tmp_trs_number = [1] * len(label.numpy().tolist()[0])
        list_messages = cut_list(messages, 50)
        rs = []
        for lms in list_messages:
            ones = {}
            ones["role"] = "user"
            ######################################################################
            if dataset == 'DD':
                # IEMOCAP
                # ones["content"] = "Predicting the emotion of the above utterances. Each utterance is one of the emotions: 'happy', 'sad', 'neutral', 'angry', 'excited', or 'frustrated'."
                # DD
                ones["content"] = "Predicting the emotion of the above utterances. Each utterance is one of the emotions: 'happiness', 'neutral', 'fear', 'disgust', 'anger', 'surprise', 'sadness'."
            else:
                # MELD
                if dataset == 'MELD3':
                    ones["content"] = "Predicting the sentiment of the above utterances. Each utterance is one of the sentiments: 'positive', 'negative', 'neutral'."
                else:
                    ones["content"] = "Predicting the emotion of the above utterances. Each utterance is one of the emotions: 'sadness', 'joy', 'neutral', 'surprise', 'anger', 'disgust', 'fear'."
                # if dataset == 'EmoryNLP3':
                #     ones["content"] = "Predicting the sentiment of the above utterances. Each utterance is one of the sentiments: 'positive', 'negative', 'neutral'."
                # else:
                #     ones["content"] = "Predicting the emotion of the above utterances. Each utterance is one of the emotions: 'joyful', 'mad', 'peaceful', 'neutral', 'sad', 'powerful', 'scared'."
            ######################################################################
            lms.append(ones)
            print(lms)
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=lms
            )
            tmp = completion.choices[0].message
            if 'choices' in tmp:
                rs += tmp['choices']['content'].split('\n')
            else:
                rs += tmp['content'].split('\n')
        print(rs)

        if dataset == 'IEMOCAP':
            for r in rs:
                if len(r) != 0:
                    for l in str_label:
                        if l in r:
                            if '.' in r and bool(re.search(r'\d', r)):
                                id = int(r.split('.')[0]) - 1
                            trs.append(l)
                            tmp_trs_number[id] = mapping[l]
                            break
        else:
            for r in rs:
                if len(r) != 0:
                    for l in str_label:
                        if l in r:
                            if bool(re.search(r'\d+[.|:]', r)):
                                id = int( re.search(r'\d+[.|:]', r).group(0)[:-1] ) - 1
                                trs.append(l)
                                tmp_trs_number[id] = mapping[l]
                                break

        trs_number += tmp_trs_number
        avg_accuracy = round(accuracy_score(labels, trs_number, sample_weight=masks) * 100, 2)
        avg_fscore = round(f1_score(labels, trs_number, sample_weight=masks, average='weighted') * 100, 2)
        print(avg_accuracy, avg_fscore)

    avg_fscore2 = f1_score(labels, trs_number, sample_weight=masks, average=None)
    print(avg_fscore2)

        # avg_accuracy = round(accuracy_score(labels, trs_number, sample_weight=masks) * 100, 2)
        # avg_fscore1 = round(f1_score(labels, trs_number, sample_weight=masks, average='micro', labels=[0, 2, 3, 4, 5, 6]) * 100, 2)
        # avg_fscore2 = round(f1_score(labels, trs_number, sample_weight=masks, average='macro') * 100, 2)
        # fscores = [avg_fscore1, avg_fscore2]
        # print(avg_accuracy, fscores)

