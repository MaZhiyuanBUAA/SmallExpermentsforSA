#coding=utf-8
import numpy as np
from loadData import loadData,dialog2vec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
global TrainPath,TestPath
TrainPath = '../data/train.txt'
TestPath = '../data/test.txt'
 
def train():
    dialogs,labels = loadData(TrainPath)
    X = dialog2vec(dialogs)
    clf = GradientBoostingClassifier(learning_rate = 0.06 ,n_estimators =100,max_features = 'log2',min_samples_leaf=2)
    clf.fit(X,labels)
    joblib.dump(clf,'../model/gbdt.m',compress=3)
    
def test():
    clf=joblib.load('../model/gbdt.m')
    dialogs,labels = loadData(TestPath)
    X = dialog2vec(dialogs)
    pred = np.argmax(clf.predict_proba(X),axis=1)
    Y = np.array(labels)
    numR = (pred-Y).tolist().count(0)
    f = file('../data/result.txt','w')
    for ind,ele in enumerate(dialogs):
        f.write('dialog:%d,sentimentScore:%d,realScore:%d'%(ind,pred[ind],Y[ind])+'\n')
        f.write('line:'.join(ele))
        f.write('##################################################################\n')
    f.close()
    print pred.tolist()
    print '\n'
    print Y.tolist()
    print 1.*numR/len(labels)

    
if __name__=='__main__':
    #train()
    test()