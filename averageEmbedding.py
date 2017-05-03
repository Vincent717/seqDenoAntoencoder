# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.3.17
# this code compute vector representation of each files, by average their word embeddings
# and find the most similar documents by computing their cosine distance with key vector

import os
import os.path
import gensim
import numpy as np
#import hashlib


#### please specify some important arguments below

# specify where the model is, and which model you are going to load 
model_path = '/home/vincent/tmp/word_vector_model/mymodel910_size200_insentence'

# spicify the folder that holds documents that are represented as list of (sentences of) words
dirpath = '/home/vincent/tmp/out1000_sentence'

# the key word that you want to find
key_word = '制造业'



def computeAverageEmbedding(model, words):
    '''
    itype: word2vec model, list[list[str]]
    rtype: array
    '''
    size = len(model['的'])
    result = np.zeros(size)
    n = 1
    for sentence in words:
        for word in sentence:
            try:
            #print(word,model[word])
                result += np.array(model[word])
                n += 1
            except KeyError:
                continue
    #print(result/n)
    return result/n


def iterateFolder(model, dirname):
    '''
    iterate the files in dirname, use model to compute its average embedding
    return a file list, and a list of document_vectors
    itype: word2vec model, str
    rtype: (resul)
    '''
    print('starting reading files ...')
    print('total in ', str(len(os.listdir(dirname))))
    i = 0
    files = os.listdir(dirname)
    result_vectors = []

    for filename in files:
        full_name = os.path.join(dirname,filename)
        print('handling ', str(i), ' of ', full_name)
        tmp_file = open(full_name,'r')
        words = tmp_file.read()
        words = eval(words)
        tmp_file.close()
        phrase_vector = computeAverageEmbedding(model,words)
        i += 1
        #s = str(phrase_vector)
        #m = hashlib.md5()
        #m.update(s.encode('utf8'))
        #result[m.hexdigest()] = filename
        result_vectors.append(phrase_vector)
    return files,result_vectors

def main():
    model = gensim.models.Word2Vec.load(model_path)
    savepath = 'phrase_vectors'
    try:    # this part is buggy, since "array" is not evaluable.
        print('tring to load phrase vectors...')
        with open(savepath+'_files','r') as f:
            files = eval(f.read())
        phrase_vectors = np.load(savepath)
    except Exception as e:
        print(repr(e))
        print('phrase vectors loading fail, now try to compute them')
        print('start iterating files in ', dirpath)
        files,phrase_vectors = iterateFolder(model, dirpath)
        print('Phrased vectors have been bulit!')

        # save the result phrase_vectors_dict
        

        file = open(savepath+'_files','w')
        file.write(str(files))
        file.close()
        result_vectors = np.array(phrase_vectors)
        np.save(savepath,result_vectors)

    if not model.__contains__(key_word):
        print('Key word not exsited, please try another key word')
    else:
        key_word_vector = model[key_word]

    #model.similar_by_vector()
    #cosines = np.array([])
    phrase_matrix = np.array(phrase_vectors)
    #print(phrase_matrix.shape)
    #print(key_word_vector.shape)
    #
    #print(cosines)
    return (files,phrase_matrix,key_word_vector)

def printResult(files,phrase_matrix,key_word_vector):
    '''
    show the results
    '''
    print('The key word is ', key_word)
    print('We have %d files in total'%len(files))
    cosines = np.dot(phrase_matrix,key_word_vector)
    flag = np.sort(cosines)[-10]
    top10 = np.where(cosines>flag)[0]
    print('The 10th has similarity ',flag)
    print('Top 10 files are: ')
    for i in top10:
        print(files[i])
    print('finish')

if __name__ == "__main__":
    result = main()
    printResult(result[0],result[1],result[2])
    #print 