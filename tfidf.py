# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.3.15

'''
building tfidf model
'''

import numpy as np
import os
import gensim
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib



#syn_set = ['wordnet', 'AI', 'artificial intelligence', 'computer vision',
#           'fuzzy logic', 'machine translation', 'neural network', 'robotics',
#           '词网','人工智能','计算机视觉','模糊逻辑','机器翻译','神经网络','机器']

syn_set = ['人工智能','计算机','逻辑','神经网络','机器','智能','学习']
#syn_set = ['教育','教诲','教导','教训','训练','机构','学习','学校','大学','学位','想法','知识','判断',
#           '课程','家教','学分','自学','上学','高等教育','初等教育','中学','小学','作业','教室','学生']
#syn_set = ['制造业','工业','制造','工厂','机器','机械','精细','过程','商品','加工','操作']
keyword = '服务业'

wordspath = 'out1000/'

def constructCorpus(dirname):
    '''
    bulid a corpus from txt file in dirpath
    itype: str
    otype: list (each document as a str in it)
    '''
    #corpus=["我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
    #   "他 来到 了 网易 杭研 大厦",#第二类文本的切词结果
    #   "小明 硕士 毕业 与 中国 科学院",#第三类文本的切词结果
    #   "我 爱 北京 天安门"]#第四类文本的切词结果
    
    i = 0
    files = os.listdir(dirname)
    corpus = []
    print('Start constructing corpus...')
    for filename in files:
        full_name = os.path.join(dirname,filename)
        print('handling ', str(i), ' of ', full_name)
        with open(full_name) as f:
            li = f.read()
            words = eval(li)
        corpus.append(' '.join(words))
        i += 1
    print('Corpus is constructed successfully!')
    return (files,corpus)



def tfidfModelBuilding(corpus):
    '''
    build tfidf model
    itype: list[str]
    rtype: (array, matrix)
    '''
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    words=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    #for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #    print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------")
        #for j in range(len(word)):
        #   print(word[j],weight[i][j])
        #print('Top 10 most frequent words for',i)
        #tmp = weight[i][j]
        #tmp.sort()
        #print max()
    joblib.dump(weight,'tfidf_model/tfidf.m')
    joblib.dump(words,'tfidf_model/words')
    print('Model is built!')
    return (words, weight)


def getTopkwords(files,words,weights,k):
    '''
    find top k words for each files, sorted by tfidf
    itype: list[str], list[str], matrix, int
    rtype: list[list[tuple(str,float)]]
    '''
    topkwords_list = []
    total_scores = []
    for i in range(len(weights)):
        tmp = np.sort(weights[i])
        flag = tmp[-k-1]     # choose top k key words
        #print('Top ',k,' frequent words of %d :'%i)
        top_arg = np.where(weights[i]>flag)[0]
        topwords = [(words[j],weights[i][j]) for j in top_arg]
        topkwords_list.append(topwords)
    return topkwords_list


def standardTfidf(files, words, weights, k):
    '''
    since we had topwords_list, we can filter out the most revelant files based on whether its top words are in synset
    itype: list[str], list[list[tuple(str,float)]], int
    rtype: none
    '''

    topwords_list = getTopkwords(files,words,weights,k)

    total_top_keys = []
    total_scores = []
    # find top 10 files with highest sum of revelant tfidf scores
    for topwords in topwords_list:
        top_key = []
        total_score = 0
        for word,score in topwords: # see whether these k words are in synset, if it is, store it
            if words in syn_set:
                top_key.append(word)
                total_score += score
        total_top_keys.append(top_key)
        total_scores.append(total_score)

    # sort total_scores and filter out top k
    total_scores = np.array(total_scores)
    total_score_flag = np.sort(total_scores)[-k-1]   # choose top k files with highest total score
    topk = np.where(total_scores>total_score_flag)[0]
    print('Top ',k,' files with highest total score')

    # print the top
    print(topk,111)
    for i in topk:
        print(files[i],total_top_keys[i],total_scores[i],111)


def tfidfWithWord2vec(files, words, weights, k, word2vec_model_path, keyword):
    '''
    the idea is, find top k tfidf words fo each files,
    and calculate cosine between the average sum of them and the target word vector
    itype: list[list[tuple(str,float)]], str
    rtype: 
    '''

    topwords_list = getTopkwords(files,words,weights,k)
    #print(topwords_list[0][0][0])

    file_vectors = []
    model = gensim.models.Word2Vec.load(word2vec_model_path)
    size = len(model[topwords_list[0][0][0]])
    # get key word vector
    try:
        key_word_vector = model[keyword]
    except:
        print('there is no such key word in this model, try another key word please')
        return
    # construct file matrix
    for file in topwords_list:
        file_vector = np.zeros(size)
        for word,weight in file:
            try:
                file_vector += model[word]
            except:
                continue
        file_vectors.append(file_vector)
    phrase_matrix = np.array(file_vectors)
    cosines = np.dot(phrase_matrix,key_word_vector)

    # find top 10 files
    flag = np.sort(cosines)[-10]
    top10 = np.where(cosines>flag)[0]
    print('The 10th has similarity ',flag)
    print('Top 10 files are: ')
    for i in top10:
        print(files[i], cosines[i])
    print('finish')



def main():
    tfidf_model_path = 'tfidf_model/tfidf.m'
    tfidf_dict_path = 'tfidf_model/words'
    try:
        print('trying to load pretrained model...')
        # see if there if already a model
        weights = joblib.load(tfidf_model_path)
        words = joblib.load(tfidf_dict_path)
        files = os.listdir(wordspath)
    except:
        print('There is no existed model, trying to train a model now...')
        files, corpus = constructCorpus(wordspath)
        words, weights = tfidfModelBuilding(corpus)
    
    k = 20
    word2vec_model_path = 'word_vector_model/mymodel910_size200_insentence'
    standardTfidf(files, words, weights, k)
    #tfidfWithWord2vec(files, words, weights, k, word2vec_model_path, keyword)


if __name__ == "__main__":
    main()
    #print 