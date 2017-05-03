# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.3.16
# this code generate the word embedding model from corpus, and save it

import os
import os.path
import gensim
import json

class MySentences(object):

    i = 0

    def __init__(self, dirname):
        self.dirname = dirname
        self.files = os.listdir(self.dirname)

    def purify(self, words):
        return words

    def __iter__(self):
        '''
        bulid a corpus from txt file in dirpath
        itype: str
        otype: list (each document as a list of words in it)
        '''
        #print('starting reading files ...')
        #print('total in ', str(len(os.listdir(self.dirname))))
        #return self

    #def __next__(self):
        
        for filename in os.listdir(self.dirname):
            #if self.i >= len(self.files):
            #    raise StopIteration()
            #filename = self.files[self.i]
            full_name = os.path.join(self.dirname,filename)
            print('handling ', str(self.i), ' of ', full_name)
            tmp_file = open(full_name,'r')
            words = tmp_file.read()
            words = eval(words)
            tmp_file.close()
            words = self.purify(words)
            self.i += 1
            #yield words

            for sentences in words:
                if sentences != []:
                    #print('processing',sentences[0])
                    yield sentences


def iterateFiles(dirname):
    print('starting reading files ...')
    print('total in ', str(len(os.listdir(dirname))))
    i = 0
    result = []
    for filename in os.listdir(dirname):
        full_name = os.path.join(dirname,filename)
        print('handling ', str(i), ' of ', full_name)
        tmp_file = open(full_name,'r')
        words = tmp_file.read()
        words = eval(words)
        tmp_file.close()
        #words = self.purify(words)
        i += 1
        result.append(words)
    #print('Going to return a ',len(words))
    return result  


def modelBuilding(sentences, savepath):
    '''
    model building
    min_count : the least number of occurence that will be considered
    size      : size for neural network
    workers   : parallelization, but Cython is needed
    '''
    model = gensim.models.Word2Vec(sentences, min_count=5, size=200)
    model.save(savepath)


    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(savepath+'_vocab', 'w') as f:
        f.write(json.dumps(vocab))
    #new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

    # online training
    #model = gensim.models.Word2Vec.load('/tmp/mymodel')
    #model.train(more_sentences)


def main():
    print('---start---')
    #modelpath = '/home/vincent/tmp/mymodel'
    # model for gonggao
    #modelpath = 'word_vector_model/mymodel910_size200_insentence'
    # model for english books
    modelpath = 'word_vector_model/english_size200_insentence'
    try:
        print('try to load model from', modelpath)
        model = gensim.models.Word2Vec.load(modelpath)
    except:
        #dirname = '/home/vincent/tmp/tiny_lab/out'
        # dir for gonggao
        #dirname = 'tmp/out1000_sentence'
        # dir for english books
        dirname = 'books_in_sentences/sep'
        print('No existed model.')
        print('Start training model from corpus in ', dirname)
        sentences = MySentences(dirname)
        #sentences = iterateFiles(dirname)
        modelBuilding(sentences, modelpath)

    # get vector
    #model['computer']

if __name__ == "__main__":
    main()