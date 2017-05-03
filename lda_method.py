# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.3.28

'''
building LDA model
'''

import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import lda



### specify some arguments
n_topics = 50
n_iter = 500
dirname = 'out1000'
#target_word = '人工智能'
target_word = '教育'


useless_words = ['适用', '其他', '公司', '投资', '资产', '有限公司', '人民币', '营业', '股东', '报告'
                ,'情况', '价值', '金额', '项目', '负债', '损益', '集团', '亿元', '有限责任', '董事会'
                ,'股份', '相关', '销售', '千元', '日止', '合计', '单位', '金额', '子公司', '财务报表'
                ,'披露', '业务', '募集', '管理', '说明', '收入', '收益', '贷款', '资本', '企业', '半年度'
                ,'万元', '本期', '年度报告', '董事', '现金', '附注', '截至', '确认', '准备', '本行'
                ,'发行', '原因', '上年', '资金', '收益', '是否', '公允', '坏账', '账面', '期末余额'
                ,'上市', '权益', '股权', '合并', '计入', '增减', '同期', '金融资产', '发行人', '公开'
                ,'应收', '债券', '风险', '计量', '金融', '風險', '資產', '貸款', '限售', '集团股份'
                ,'当期', '变动', '垫款', '币种', '上半年', '百万元', '增长', '期初余额', '计提'
                ,'经营', '交易', '支付', '成本', '减值', '持有', '处置', '行业', '产品', '生产', '主要'
                ,'应收款', '活动' ,'所得税', '主要', '减少', '持股', '增加', '发生', '进行', '经营', '公告'
                ,'关于', '审计', '報告', '本集團', '投資', '费用', '会计', '长期', '取得', '处置', '注明'
                ,'比例', '借款', '中国', '账款', '期间', '出售', '损失', '承诺', '公积', '收到', '合計'
                ,'財務', '億元', '期末', '产生', '百分点', '实现', '未知', '余额', '期内', '变更', '浮动'
                ,'北京市', '北京', '上海', '广东', '重庆', '重庆市', '山东省', '山东', '四川', '四川省'
                ,'上海市', '南京', '杭州', '宁波', '常州', '吉林', '黑龙江省', '武汉', '新疆', '香港', '深圳'
                ,'山西', '西安', '陕西', '湖南', '广西', '洛阳', '大连', '广州', '青岛' 
                ,'综合', '其中', '现金流量', '固定资产', '关联', '应付', '上期', '确定', '无形资产'
                ,'按照', '净额', '报告期', '现金流量', '年度', '上市公司', '数量', '回购', '条件', '公司股票', '条件'
                ,'投入', '理财', '委托', '本人', '减持', '首次', '所有者', '使用', '授予', '综合', '其中', '控制'
                ,'担保', '事项', '本期发生', '上期', '组合', '股票', '计划', '按照', '股本', '及其', '净资产'
                ,'母公司', '否否', '上述'
                ]

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
        #print('handling ', str(i), ' of ', full_name)
        with open(full_name) as f:
            li = f.read()
            words = eval(li)
        corpus.append(' '.join(words))
        i += 1
    print('Corpus is constructed successfully!')
    return corpus, files


def ldaModelBuilding(corpus, n_topics=50, n_iter=500):  #50,100
    '''
    build LDA model
    itype: list[str], int, int
    rtype: model
    '''
    print('Start building model...')
    vectorizer=CountVectorizer(stop_words=useless_words)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()    # 词频矩阵
    words = vectorizer.get_feature_names()#获取词袋模型中的所有词语

    print("type(X): {}".format(type(X)))
    print("shape: {}\n".format(X.shape))

    # build and train
    model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
    model.fit(X)

    # save model
    model_name = 'ldamodel_topic'+ str(n_topics) + '_iter' + str(n_iter)
    joblib.dump(model, 'lda_model/'+model_name)
    joblib.dump(words, 'lda_model/'+model_name+'_words')


    # topic-word distribution
    topic_word = model.topic_word_
    print("type(topic_word): {}".format(type(topic_word)))
    print("shape: {}".format(topic_word.shape))

    # get top 5 words in each topic
    # beautiful code
    n = 10
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(words)[np.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
    print('model saved!')
    return model, words



def standardLda(model, words, files, target_word):
    '''
    see the probablity of generating the target word from max topic for each doc
    itype: model
    rtype: 
    '''

    print('Begin to print result...')
    getTopicWords = lambda x: np.array(words)[np.argsort(topic_word[x])][:-6:-1] 
    
    # document topic distribution
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    # topic-word distribution
    topic_word = model.topic_word_

    file_probs = []
    # get the most possible topic for each doc
    for n in range(len(files)):
        topic_most_pr = doc_topic[n].argmax()
        topic_words = getTopicWords(topic_most_pr)    # top 5 words under topic_most_pr topic
        #print("doc: {} topic: {} roughly like: {}".format(n, topic_most_pr, ' '.join(topic_words)))
        # the probability of target_word among the most probable topic of this file
        prob = topic_word[topic_most_pr][words.index(target_word)]
        file_probs.append(prob)

    # print top 10 documents
    print('top 10 documents for ', target_word)
    file_probs = np.array(file_probs)
    for i in np.argsort(file_probs)[:-(10+1):-1]:
        print(files[i], file_probs[i], getTopicWords(doc_topic[i].argmax()))


def main():
    # get model: load or train
    print('Trying to load the model...')
    
    try:
        model_name = 'ldamodel_topic'+ str(n_topics) + '_iter' + str(n_iter)
        model_path = 'lda_model/' + model_name
        model_words_path = 'lda_model/' + model_name + '_words'
        model = joblib.load(model_path)
        words = joblib.load(model_words_path)
        files = os.listdir(dirname)
        print('model loading successfully! From ',model_path)
    except:
        corpus, files = constructCorpus(dirname)
        model, words = ldaModelBuilding(corpus, n_topics=n_topics, n_iter=n_iter)

    # show me the result
    standardLda(model, words, files, target_word)



if __name__ == "__main__":
    main()
