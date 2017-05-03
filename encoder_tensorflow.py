# -*- coding: UTF-8 -*-

# __author__ = Huang Wenguan
# date  : 2017.4.7

'''
building my final model on keras
'''

from __future__ import print_function
import numpy as np

from random import random
import os
import json
import time
import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate, dot, pooling, RepeatVector, merge
from keras.utils import to_categorical
from keras.layers.wrappers import TimeDistributed
from keras import backend as K


dirname = 'out1000_sentence_try'
maxlen = 22
embeddings_path = 'word_vector_model/mymodel910_size200_insentence.wv.syn0.npy'
vocab_path = 'word_vector_model/mymodel910_size200_insentence_vocab'
save_path = 'keras_model/bilstm_maxpool_lstm_gonggao910'

# My model

#class myEncodeModel():


def load_vocab(vocab_path):
    '''
    loading word2vec vocab
    '''
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def corpusConstructing(dirname, maxlen, tag='itself',p1=0.1,p2=0.1, **params):
    '''
    training data shape should be: (batch, len(sentence)) 
    training label shape should be: (batch, len(sentence))
    tag = 'itself' : require p1, p2 to noising the original sentence a little bit, and return it as x
    tag = 'adjacent': the next sentence. (maybe also consider the previous sentence later)
    tag = 'both': the above two
    itype: dirname: str
           maxlen : int
           tag    : str
    rtype: list[list[str]]  
    '''
    filterp1 = lambda li, p1 : [i for i in li if random() > p1]
    def switchp2(li, p2):
        for i in range(1,len(li)):
            if random() < p2:
                li[i-1],li[i] = li[i],li[i-1]
        return li

    i = 0
    files = os.listdir(dirname)
    corpus = []
    print('Start constructing corpus...')
    for filename in files:
        full_name = os.path.join(dirname,filename)
        #print('handling ', str(i), ' of ', full_name)
        with open(full_name) as f:
            li = f.read()
            sentences = eval(li)
        # sentences.append(['0'] * maxlen) WE need this when it is adjacent
        corpus += sentences
        i += 1
    corpus = [sent for sent in corpus if sent != []]
    print('Corpus is constructed successfully! You have %d sentence from %d files'%(len(corpus), len(files)), )

    # change word in index in vocab
    word2idx, idx2word = load_vocab(vocab_path)
    index_corpus = [[word2idx[word] if word in word2idx else 0 for word in sent] for sent in corpus]

    if tag == 'itself':
        y_train1 = index_corpus
        x_train = [filterp1(sent,p1) for sent in index_corpus]
        x_train = [switchp2(sent,p2) for sent in x_train]
    if tag == 'adjacent':
        x_train = index_corpus
        y_train1 = index_corpus[1:].append(['0']*maxlen)
    #if tag == 'both':
    
    # saving data
    with open(save_path+'_data_x', 'w') as f:
        f.write(json.dumps(x_train))
    with open(save_path+'_data_y', 'w') as f:
        f.write(json.dumps(y_train1))

    print('returning corpus...')
    return x_train, y_train1



def modelBuiliding(x_train, y_train, steps_per_epoch=30,epochs=3, tag = 'itself', extraEmbed = True,
                    max_features=20000,d=200,u=100,da=200,maxlen=22,r=20,dicsize=1000, **params):
    '''
    building model
    itype:
        tag = 'itself': only use one output, that is itself
        extraEmbed : (Bool) whether use pretrain extra embedding weights
        max_features : (Int) 
        d : dimension of embedding
        u : dimension of lstm of encoder
        da: dimension of weight1 of attention mechanism
        r : number of attentions
        maxlen : the max lenght of sentence, those who behind that will be cut and not considered
        dicsize: dictionary max size
    rtype: model
    '''
    print('start building model...')

    seeOutputShape = lambda x : Model(inputs=encode_input,outputs=x).output_shape
    seeInputShape = lambda x : Model(inputs=encode_input,outputs=x).input_shape

    ## Input(shape=None, batch_shape=None, name=None, dtype='float32', sparse=False, tensor=None)
    # shape: A shape tuple (integer), not including the batch size.
    encode_input = Input(shape=(maxlen,), name='main_input')

    ## Embedding(vocabulary_size, embedding_dimension, input_length=max_len)
    #model = Model(inputs=encode_input, outputs=encode_embedlayer)
    # http://ben.bolte.cc/blog/2016/keras-gensim-embeddings.html
    if extraEmbed:
        weights = np.load(open(embeddings_path, 'rb'))
        dicsize = weights.shape[0]
        encode_embedlayer = Embedding(input_dim = weights.shape[0], output_dim=weights.shape[1], weights=[weights], input_length = maxlen)(encode_input)
    else:
        encode_embedlayer = Embedding(input_dim = dicsize, output_dim=d, input_length = maxlen)(encode_input)
    

    ## LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
    encode_lstm_left = LSTM(u,return_sequences=True)(encode_embedlayer)
    encode_lstm_right = LSTM(u,return_sequences=True,go_backwards=True)(encode_embedlayer)
    encode_hiddenlayer_out = concatenate([encode_lstm_left, encode_lstm_right],axis=-1)
    
    ### attention
    ### another tricky bug
    ## ValueError: Input dimension mis-match. (input[0].shape[1] = 50, input[1].shape[1] = 350)
    ## Inputs shapes: [(1, 50, 350), (1, 350, 1)]
    ## so (1,50,350) is what I gave him, while (1,350,1) is what he excepted (Dense)
    ## since Dense do not support matrix input, so we need timeDistributed wrapper
    def penalty(weight_matrix):
        AAT = K.batch_dot(weight_matrix, weight_matrix, axes=[1,1])
        AAT = AAT - K.eye(r)
        #K.update_sub(AAT,K.eye(r))   #K.int_shape(AAT)[1])
        return 0.01 * (K.sum(K.square(AAT)))
    attention_hidden = TimeDistributed(Dense(da, input_shape=(maxlen,2*u), activation='tanh'))(encode_hiddenlayer_out) # input_shape = (None, n, 2u)
    attention_out = TimeDistributed(Dense(r, input_shape=(maxlen,da), activation='softmax', activity_regularizer = penalty))(attention_hidden)
    #attention_out = TimeDistributed(Dense(r, input_shape=(maxlen,da), activation='softmax'))(attention_hidden)

    #encode_out = merge([attention_out, encode_hiddenlayer_out], mode='dot', dot_axes=0)
    encode_out = dot([attention_out, encode_hiddenlayer_out], axes=1, name='dot_of_attention') # output_shape = r*2u


    ### decoder
    decode_input = pooling.MaxPooling1D(pool_size=r, strides=None, padding='valid')(encode_out)

    ## here is a tricky part, we need to specify the input of the decoder, by repeating maxlen times paravector
    # also, since the output_shape after maxpooling is (None, 1, 2*u), what we require is (None, 2*u),
    # we cannot use RepeatVector, but use concatenate instead
    # however, we still want to try the standard seq2seq decoder setting 
    #decode_input_repeat = RepeatVector(maxlen)(decode_input)
    decode_input_repeat = concatenate([decode_input]*maxlen, axis=1)


    # if dicsize is too large, it suffers from memory error
    decode_hidden_itself = LSTM(2*u, return_sequences=True)(decode_input_repeat)
    decode_output_itself = TimeDistributed(Dense(dicsize, activation='softmax', name='output_itself'))(decode_hidden_itself)
    #decode_ouput_adjacent = LSTM(dicsize, return_sequences=True, activation='softmax', name='output_adjacent')(decode_input_repeat)


    #decode_ouput_itself = Dense(1, activation='softmax', name='output_itself')(decode_hiddenlayer_out)
    #decode_ouput_adjacent = Dense(1, activation='softmax', name='output_adjacent')(decode_hiddenlayer_out)

    
    model = Model(inputs=encode_input, outputs=decode_output_itself) #, decode_ouput_adjacent])
    encoder = Model(inputs=encode_input, outputs=encode_out)
    attention = Model(inputs=encode_input, outputs=attention_out)

    print('Model build successfully!')
    print('output shape: ', model.output_shape, '\ninput shape: ',model.input_shape)


#def modelTraining(model,x_train, y_train, save_path, steps_per_epoch=30, epochs=100):
    '''
    
    '''
    # try using different optimizers and different optimizer configs
    # np.expand_dims(y,-1)

    # print('Loading data...')

    # x_train = np.random.randint(dicsize,size=(2000,maxlen))
    # y_train1 = x_train
    # y_train2 = np.random.randint(dicsize,size=(2000,maxlen))
    # x_test = np.random.randint(dicsize,size=(500,maxlen))
    # y_test1 = x_test
    # y_test2 = np.random.randint(dicsize,size=(500,maxlen))

    # print(len(x_train), 'train sequences')
    # print(len(x_test), 'test sequences')


    def generate_arrays(x_train, y_train):
        '''
        ### use fit_generator instead, to avoid OutOfMemory Error
        ## a really tricky bug, 
        #ValueError: Error when checking model input: expected main_input to have shape (None, 50) but got array with shape (50, 1)
        # slove it by warpping a empty warper of the input, letting it change from (50,) to (1, 50), which then maps with (None, 50)
        # Convert labels to categorical one-hot encoding
        tag = 'itself': using 
        tag = 'both' : using 
        '''
        while 1:
            for i in range(len(x_train)):
                #yield ({'main_input': np.array([x_train[i]])} , {'output_itself': np.array([to_categorical(y_train[i],1000)])}) 
                                                     #,'output_adjacent': np.array([to_categorical(y_train2[i],1000)])})
                yield (np.array([x_train[i]]), np.array([to_categorical(y_train[i],dicsize)])) 
                #yield (x_train[i], to_categorical(y_train[i],dicsize)) 


    print('Compiling...')
    time_start = time.time()
    myadam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer= myadam,
                  #exception_verbosity='high',
                  #optimizer=fast_compile
                  loss='categorical_crossentropy',
                  #loss_weights=[1,0.5],
                  metrics=['accuracy'])
    time_end = time.time()
    print('Compiled, cost time:%f second!' % (time_end - time_start))


    print('Training...')

    time_start = time.time()
    ## padding 0 in the beginning of the sequence, to reach the maxlen
    print("Pad sequences (samples x time)")
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    y_train = sequence.pad_sequences(y_train, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)


    model.fit_generator(generate_arrays(x_train,y_train),  #1,y_train),
            steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    time_end = time.time()
    print('Trained, cost time:%f second!' % (time_end - time_start))

    save_path_final = save_path + '_d'+str(d)+'_u'+str(u)+'_da'+str(da)+'_r'+str(r)
    save_path_encoder = save_path_final + '_encoder'
    save_path_attention = save_path_final + '_attention'
    model.save(save_path_final)
    encoder.save(save_path_encoder)
    attention.save(save_path_attention)
    print('training finished!')

    return model


def main():
    try:
        print('Try loading data...')
        with open(save_path+'_data_x', 'r') as f:
            x_train = json.loads(f.read())
        with open(save_path+'_data_y', 'r') as f:
            y_train1 = json.loads(f.read())
        print('Data loading succeed! We have %d sentence'%len(x_train))
    except:
        print('Data loading fail, try to construct data set...')
        x_train, y_train1 = corpusConstructing(dirname, maxlen,p1=0.1, p2=0.1)
    
    model = modelBuiliding(x_train,y_train1,steps_per_epoch=64,epochs=30000)


if __name__ == "__main__":
    main()
    #print 
