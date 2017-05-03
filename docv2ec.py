# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.4.02

'''
building doc2vec model
'''

import os
import numpy as np
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models import Word2Vec

# some parameters
dirname = 'out1000'
size = 200
modelname = 'doc2vec_model/doc2vec_size'+str(size)
#target_word = '人工智能'
target_word = '制造业'
word_model_name = 'word_vector_model/mymodel910_size200_insentence'

class LabeledDocSentence(object):
    '''
    A wrapper for LabeldSentence class
    # sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])
    Doc2vec接受一个由LabeledSentence对象组成的迭代器作为其构造函数的输入参数。
    其中，LabeledSentence是Gensim内建的一个类，它接受两个List作为其初始化的参数：word list和label list。
    '''
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for uid, filename in enumerate(os.listdir(self.dirname)):
            full_name = os.path.join(self.dirname,filename)
            with open(full_name) as f:
                li = f.read()
                words = eval(li)
            print(uid,filename)
            yield LabeledSentence(words=words, tags=['DOC_%s' % uid])
            


def doc2vecModelBulding(dirname, filename, size):
    '''
    bulding model, using corpus in dirname, and save the model in filename
    itype: str,str
    rtype: model
    '''
    # dm defines the training algorithm. By default (dm=1), ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
    # alpha is the initial learning rate (will linearly drop to zero as training progresses).
    # workers = use this many worker threads to train the model (=faster training with multicore machines).
    # hs = if 1 (default), hierarchical sampling will be used for model training (else set to 0).
    # negative = if > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
    
    documents = LabeledDocSentence(dirname)

    model = Doc2Vec(documents, dm=1, size=size, window=8, negative=10, hs=0, min_count=4, workers=4)
    model.save(filename)
    #model = Doc2Vec.load(filename)
    return model


def getWordVector(target_word):
    modelpath = 'word_vector_model/mymodel910_size200_insentence'
    model = Word2Vec.load(modelpath)
    try:
        target_vector = model[target_word]
        return target_vector
    except Exception as e:
        print(repr(e))

def printResult(target_word, model, dirname):
    '''
    print top10 files whose paravector has the largest consine similarity with target_vector
    itype: np.array, model, str
    rtype:
    '''
    files = os.listdir(dirname)
    docvecs = model.docvecs
    cosines = []
    try:
        #target_vector = model[target_word]
        target_vector = getWordVector(target_word)
        cosines = [np.dot(target_vector,docvec) for docvec in docvecs]
        cosines = np.array(cosines)
        for i in np.argsort(cosines)[:-(10+1):-1]:
            print(files[i], cosines[i])
    except Exception as e:
        print(repr(e))
        print('target word no existed, please change another one')



def main():
    try:
        print('trying to load model...')
        model = Doc2Vec.load(modelname)
    except:
        print('Loading failed, trying to train a model...')
        model = doc2vecModelBulding(dirname,modelname,size)

    #wordmodel = Word2Vec.load(word_model_name)
    #return model, wordmodel
    printResult(target_word,model,dirname)

if __name__ == "__main__":
    main()

