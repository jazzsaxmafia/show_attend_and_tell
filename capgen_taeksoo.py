# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import ipdb
import cPickle

from keras.preprocessing import sequence
from keras import activations, initializations
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix

#dictionary = pd.read_pickle('/home/taeksoo/Study/Multimodal/dataset/flickr30/dictionary.pkl')
#dictionary[0] = '#START#'
#dictionary[1] = '.'
#n_vocab = len(dictionary)
#with open('/home/taeksoo/Study/Multimodal/dataset/flickr30/flicker_30k_align.train.pkl') as f:
#    sentences = cPickle.load(f)
#    image_feats = cPickle.load(f)
#
#sentences, image_ids = zip(*sentences)
#sent_to_num = map(lambda sent: map(lambda word: dictionary[word] if word in dictionary else None, sent.lower().split(' ')), sentences)
#sent_to_num = map(lambda sent: filter(lambda word: word is not None, sent), sent_to_num)
#
#X_train = sequence.pad_sequences(sent_to_num, padding='post')
#mask_train = np.ones_like(X_train) * (1 - np.equal(X_train, 0))

n_vocab = 18254
dim_word = 256
dim_ctx = 512
dim = 256

############# Building Models ################
x = T.imatrix() # (n_samples, n_timesteps)
mask = T.matrix(dtype='float32') # (n_samples, n_timesteps)
ctx = T.tensor3(dtype='float32') # (n_samples, 196, 512)

ctx_mean = ctx.mean(axis=1) # (n_samples, 512)

n_samples = x.shape[0]
n_timesteps = x.shape[1]

### Word Embedding ###
Wemb = initializations.uniform((n_vocab, dim_word))
emb = Wemb[x] # (n_samples, n_timesteps, dim_word)
emb_shifted = T.zeros_like(emb)
emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1]) # 맨 앞에 0을 padding함. 예측해야되니까
emb = emb_shifted

### LSTM initialization NN ###
Init_state_W = initializations.uniform((dim_ctx, dim))
Init_state_b = shared_zeros((dim))

Init_memory_W = initializations.uniform((dim_ctx, dim))
Init_memory_b = shared_zeros((dim))

initial_state = T.dot(ctx_mean, Init_state_W) + Init_state_b # (n_samples, dim)
initial_memory = T.dot(ctx_mean, Init_memory_W) + Init_memory_b # (n_samples, dim)

### Main LSTM ###
lstm_W = initializations.uniform((dim_word, dim * 4))
lstm_U = initializations.uniform((dim, dim*4))
lstm_b = shared_zeros((dim*4))

Wc = initializations.uniform((dim_ctx, dim*4)) # image -> LSTM hidden
Wc_att = initializations.uniform((dim_ctx, dim_ctx)) # image -> 뉴럴넷 한번 돌린것
Wd_att = initializations.uniform((dim, dim_ctx)) # LSTM hidden -> image에 영향
b_att = shared_zeros((dim_ctx))

U_att = initializations.uniform((dim_ctx, 1)) # image 512개 feature 1차원으로 줄임
c_att = shared_zeros((1))

emb = emb.dimshuffle(1, 0 , 2) #(n_samples, n_timesteps, dim_word) => (n_timesteps, n_samples, dim_word)
mask_shuffled = mask.dimshuffle(1,0) # (n_samples, n_timesteps) => (n_timesteps, n_samples)

### Decoding NeuralNets ###
decode_lstm_W = initializations.uniform((dim, dim_word))
decode_lstm_b = shared_zeros((dim_word))

decode_word_W = initializations.uniform((dim_word, n_vocab))
decode_word_b = shared_zeros((n_vocab))

def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:,:,n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]

def _step(m_tm_1, x_t, h_tm_1, c_tm_1, alpha_tm_1, alpha_sample_tm_1, weighed_ctx_tm_1):

    # m_tm_1 : (n_samples, 1)
    # x_t : (n_samples, dim)
    # h_tm_1 : (n_samples, dim)
    # c_tm_1 : (n_samples, dim)
    # alpha_tm_1 : (n_samples, 196)
    # alpha_sample_tm_1 : (n_samples, 196)
    # att_ctx_tm_1 :  (n_samples, 512)
    # 근데 사실상 함수 내에서 쓰이는 변수는 m_tm_1, x_t, h_tm_1, c_tm_1 뿐임.
    # 나머지는 그냥 각 step마다 return만 됨. (outputs_info에 포함되어 있어서 강제로 input에 포함)

    projected_ctx =  T.dot(ctx, Wc_att) + T.dot(h_tm_1, Wd_att)[:,None,:] + b_att # (n_samples, 196, 512)
    projected_ctx = T.tanh(projected_ctx)

    alpha = T.dot(projected_ctx, U_att) + c_att # (n_samples, 196, 1)
    alpha_shape = alpha.shape

    # 귀찮으니 일단 deterministic attention만 구현한다
    alpha = T.nnet.softmax(alpha.reshape([alpha_shape[0], alpha_shape[1]])) # 마지막 dimension 없앰
    weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (n_samples, 196, 512) * (n_samples, 196, 1)
    alpha_sample = alpha

    lstm_preact = T.dot(h_tm_1, lstm_U) + x_t + T.dot(weighted_ctx, Wc) # (n_samples, dim*4)
    i = T.nnet.sigmoid(_slice(lstm_preact, 0, dim)) # (n_samples, dim)
    f = T.nnet.sigmoid(_slice(lstm_preact, 1, dim)) # (n_samples, dim)
    o = T.nnet.sigmoid(_slice(lstm_preact, 2, dim)) # (n_samples, dim)
    c = T.tanh(_slice(lstm_preact, 3, dim)) # (n_samples, dim)

    c = f * c_tm_1 + i * c # (n_samples, dim)
    c = m_tm_1[:, None] * c + (1. - m_tm_1)[:,None] * c_tm_1 # (n_samples, dim)

    h = o * T.tanh(c) # (n_samples, dim)
    h = m_tm_1[:, None] * h + (1. - m_tm_1)[:,None] * h_tm_1 # (n_samples, dim)


    return [h, c, alpha, alpha_sample, weighted_ctx]

X_t = T.dot(emb, lstm_W) + lstm_b # (n_timesteps, n_samples, dim*4)

alpha_init = T.alloc(0., n_samples, ctx.shape[1]) # (n_samples, 196)
alpha_sample_init = T.alloc(0., n_samples, ctx.shape[1]) # (n_samples, 196)
weighted_ctx_init = T.alloc(0., n_samples, ctx.shape[2]) # (n_samples, 512)

sequences = [mask_shuffled, X_t]
outputs_info = [
    initial_state,
    initial_memory,
    alpha_init,
    alpha_sample_init,
    weighted_ctx_init
    ]

rval, updates = theano.scan(_step,
                            sequences=sequences,
                            outputs_info=outputs_info)
                            #n_steps=n_timesteps)

hiddens, cells, alphas, alpha_samples, weighted_ctxs = rval

decoded_word_vec = T.dot(hiddens, decode_lstm_W) + decode_lstm_b
decoded_word = T.dot(decoded_word_vec, decode_word_W) + decode_word_b

decoded_word_shape = decoded_word.shape
probs = T.nnet.softmax(decoded_word.reshape([decoded_word_shape[0]*decoded_word_shape[1], decoded_word_shape[2]]))

x_flat = x.flatten() # x_flat: [1   27  39  10  ...]
p_flat = probs.flatten() # p_flat: [1 => 0100000..., 27 => 00000...1000..., 39 => 00000...00100...] 이런식
cost = -T.log(p_flat[T.arange(x_flat.shape[0])*probs.shape[1] + x_flat] + 1e-8)
# x_flat.shape[0] : n_samples * n_timesteps. arange()하니까 (0 ~ n_samples*n_timesteps - 1)
# probs.shape[1] : n_vocab

cost = cost.reshape([x.shape[0], x.shape[1]])
masked_cost = cost * mask
cost = (masked_cost).sum(1)

ff = theano.function(inputs=[ctx, x, mask],
                     outputs=rval,
                     allow_input_downcast=True)

f_cost = theano.function(inputs=[ctx, x, mask],
                         outputs=masked_cost,
                         allow_input_downcast=True)

### Sampling Mode ###

