# seqDenoAntoencoder

Here is some code for tackling topic analysis task. That is, given a topic word and a bunch of text files, find the most relevant files with respect to the topic word.

I have used
* tfidf
* lad
* doc2vec (paragraph vector)
* average word embedding
* sequence denoising autoencoder

for sequence denoising autoencoder, I have implement based on keras, while the Theano version and Tensorflow version has a slightly different on the implementation of the penalty function, but basically the same.
** note that our model use pretrain word embedding, so you should either pretrain some word vectors or change the default setting in encoder_xxx.py **

To use them, just run
** python xxx.py **

It requires:
1. gensim
2. numpy
3. lda
4. keras
5. theano or tensorflow


It will print out the result.

Have fun.
