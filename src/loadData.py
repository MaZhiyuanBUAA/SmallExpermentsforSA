#coding=utf-8

import cPickle
import jieba
import numpy as np
global TrainPath
TrainPath = '../data/train.txt'
TestPath = '../data/test.txt'

def loadData(path):
    f = file(path)
    lines = f.readlines()
    f.close()
    dialogs = []
    dialog = []
    labels = []
    for line in lines:
        if line[0] == '#':
            labels.append(int(line[1]))
            dialogs.append(dialog)
            dialog = []
        else:
            dialog.append(line.split('|')[1].split(':')[1])
    dialogs.append(dialog)
    dialogs = dialogs[1:]
    labels = [1 if ele == 1 else 0 for ele in labels]
    return dialogs,labels
def getDict(dialogs):
    D = {}
    for dialog in dialogs:
        for line in dialog:
            words = jieba.cut(line)#unicode
            for word in words:
                try:
                    D[word] += 0
                except:
                    D[word] = len(D)
    f = file('../data/dict.pkl','w')
    cPickle.dump(D,f)
    f.close()
def dialog2vec(dialogs):
    f = file('../data/dict.pkl')
    D = cPickle.load(f)
    f.close()
    size = len(D)
    data = []
    for dialog in dialogs:
        vec = np.zeros(size)
        for line in dialog:
            words = jieba.cut(line)
            for word in words:
                try:
                    vec[D[word]] += 1
                except:
                    continue
        data.append(vec)
    return data
if __name__=='__main__':
    dialogs,labels = loadData(TrainPath)
    #getDict(dialogs)
    #data_x = dialog2vec(dialogs)
    
        