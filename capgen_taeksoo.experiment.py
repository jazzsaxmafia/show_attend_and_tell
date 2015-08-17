# -*- coding: utf-8 -*-
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
############# Building Models ################
class Main_model():
    def __init__(self, n_vocab, dim_word, dim_ctx, dim):
        self.trng = RandomStreams(1234)
        self.n_vocab = n_vocab
        self.dim_word = dim_word
        self.dim_ctx = dim_ctx
        self.dim = dim

        ### Word Embedding ###
        self.Wemb = initializations.uniform((n_vocab, self.dim_word))

        ### LSTM initialization NN ###
        self.Init_state_W = initializations.uniform((self.dim_ctx, self.dim))
        self.Init_state_b = shared_zeros((self.dim))

        self.Init_memory_W = initializations.uniform((self.dim_ctx, self.dim))
        self.Init_memory_b = shared_zeros((self.dim))


        ### Main LSTM ###
        self.lstm_W = initializations.uniform((self.dim_word, self.dim * 4))
        self.lstm_U = initializations.uniform((self.dim, self.dim*4))
        self.lstm_b = shared_zeros((self.dim*4))

        self.Wc = initializations.uniform((self.dim_ctx, self.dim*4)) # image -> LSTM hidden
        self.Wc_att = initializations.uniform((self.dim_ctx, self.dim_ctx)) # image -> 뉴럴넷 한번 돌린것
        self.Wd_att = initializations.uniform((self.dim, self.dim_ctx)) # LSTM hidden -> image에 영향
        self.b_att = shared_zeros((self.dim_ctx))

        self.U_att = initializations.uniform((self.dim_ctx, 1)) # image 512개 feature 1차원으로 줄임
        self.c_att = shared_zeros((1))

        ### Decoding NeuralNets ###
        self.decode_lstm_W = initializations.uniform((self.dim, self.dim_word))
        self.decode_lstm_b = shared_zeros((self.dim_word))

        self.decode_word_W = initializations.uniform((self.dim_word, n_vocab))
        self.decode_word_b = shared_zeros((n_vocab))

        self.params = [self.Wemb,
                       self.Init_state_W, self.Init_state_b,
                       self.Init_memory_W, self.Init_memory_b,
                       self.lstm_W, self.lstm_U, self.lstm_b,
                       self.Wc, self.Wc_att, self.Wd_att, self.b_att,
                       self.U_att, self.c_att,
                       self.decode_lstm_W, self.decode_lstm_b,
                       self.decode_word_W, self.decode_word_b]


    def get_initial_lstm(self, ctx_mean):
        initial_state = T.dot(ctx_mean, self.Init_state_W) + self.Init_state_b # (n_samples, dim)
        initial_memory = T.dot(ctx_mean, self.Init_memory_W) + self.Init_memory_b # (n_samples, dim)

        return initial_state, initial_memory

    def forward_lstm(self, ctx, emb, mask, initial_state, initial_memory, one_step=False):

        if one_step:
            assert initial_state, "previous state must be provided"

        if emb.ndim == 3: #(n_timesteps, n_samples, dim)
            n_samples = emb.shape[1]
        else:
            n_samples = 1


        if mask is None:
            mask_shuffled = T.alloc(1., emb.shape[0], 1)

        else:
            mask_shuffled = mask.dimshuffle(1,0) # (n_samples, n_timesteps) => (n_timesteps, n_samples)


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

            projected_ctx =  T.dot(ctx, self.Wc_att) + T.dot(h_tm_1, self.Wd_att)[:,None,:] + self.b_att # (n_samples, 196, 512)
            projected_ctx = T.tanh(projected_ctx)

            alpha = T.dot(projected_ctx, self.U_att) + self.c_att # (n_samples, 196, 1)
            alpha_shape = alpha.shape

            # 귀찮으니 일단 deterministic attention만 구현한다
            alpha = T.nnet.softmax(alpha.reshape([alpha_shape[0], alpha_shape[1]])) # 마지막 dimension 없앰
            weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (n_samples, 196, 512) * (n_samples, 196, 1)
            alpha_sample = alpha

            lstm_preact = T.dot(h_tm_1, self.lstm_U) + x_t + T.dot(weighted_ctx, self.Wc) # (n_samples, dim*4)
            i = T.nnet.sigmoid(_slice(lstm_preact, 0, self.dim)) # (n_samples, dim)
            f = T.nnet.sigmoid(_slice(lstm_preact, 1, self.dim)) # (n_samples, dim)
            o = T.nnet.sigmoid(_slice(lstm_preact, 2, self.dim)) # (n_samples, dim)
            c = T.tanh(_slice(lstm_preact, 3, self.dim)) # (n_samples, dim)

            c = f * c_tm_1 + i * c # (n_samples, dim)
            c = m_tm_1[:, None] * c + (1. - m_tm_1)[:,None] * c_tm_1 # (n_samples, dim)

            h = o * T.tanh(c) # (n_samples, dim)
            h = m_tm_1[:, None] * h + (1. - m_tm_1)[:,None] * h_tm_1 # (n_samples, dim)


            return [h, c, alpha, alpha_sample, weighted_ctx]

        X_t = T.dot(emb, self.lstm_W) + self.lstm_b # (n_timesteps, n_samples, dim*4)

        alpha_init = T.alloc(0., n_samples, ctx.shape[-2]) # (n_samples, 196)
        alpha_sample_init = T.alloc(0., n_samples, ctx.shape[-2]) # (n_samples, 196)
        weighted_ctx_init = T.alloc(0., n_samples, ctx.shape[-1]) # (n_samples, 512)

        sequences = [mask_shuffled, X_t]

        if one_step:
            rval = _step(mask_shuffled, X_t, initial_state, initial_memory, None, None, None)

        else:
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

        return rval

    def build_model(self):

        x = T.imatrix() # (n_samples, n_timesteps)
        mask = T.matrix(dtype='float32') # (n_samples, n_timesteps)
        ctx = T.tensor3(dtype='float32') # (n_samples, 196, 512)

        initial_state, initial_memory = self.get_initial_lstm(ctx.mean(axis=1))

        emb = self.Wemb[x] # (n_samples, n_timesteps, dim_word)
        emb_shifted = T.zeros_like(emb)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1]) # 맨 앞에 0을 padding함. 예측해야되니까
        emb = emb_shifted

        emb = emb.dimshuffle(1,0,2) # (n_timesteps, n_samples, dim_word)

        rval = self.forward_lstm(ctx, emb, mask, initial_state, initial_memory)

        hiddens, cells, alphas, alpha_samples, weighted_ctxs = rval

        decoded_word_vec = T.dot(hiddens, self.decode_lstm_W) + self.decode_lstm_b
        decoded_word_vec = T.tanh(decoded_word_vec)
        decoded_word = T.dot(decoded_word_vec, self.decode_word_W) + self.decode_word_b

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

        return x, mask, ctx, alphas, alpha_samples, cost

    def build_sampling_function(self):
        ctx = T.matrix()

        initial_state, initial_memory = self.get_initial_lstm(ctx.mean(axis=0))
        f_init = theano.function(inputs=[ctx],
                                 outputs=[ctx, initial_state, initial_memory])

        ctx = T.matrix()
        x = T.ivector()
        emb = T.switch(x[:,None] < 0, T.alloc(0., 1, self.Wemb.shape[1]), self.Wemb[x])

        initial_state = T.vector()
        initial_memory = T.vector()

        [
            next_state,
            next_memory,
            alphas,
            alpha_samples,
            weighted_ctxs
        ] =  self.forward_lstm(ctx,
                               emb,
                               None,
                               initial_state[None, :],
                               initial_memory[None, :],
                               one_step=True)


        hiddens = next_state

        decoded_word_vec = T.dot(hiddens, self.decode_lstm_W) + self.decode_lstm_b
        decoded_word_vec = T.tanh(decoded_word_vec)

        decoded_word = T.dot(decoded_word_vec, self.decode_word_W) + self.decode_word_b

        next_probs = T.nnet.softmax(decoded_word)
        next_sample = self.trng.multinomial(pvals=next_probs).argmax(1)

        f_next = theano.function(inputs=[x, ctx, initial_state, initial_memory],
                                 outputs=[next_probs, next_sample, next_state, next_memory],
                                 allow_input_downcast=True)

        return f_init, f_next


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):

    grads = T.grad(cost=cost, wrt=params)
    updates = []

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    return updates

def train():

    n_vocab = 18254
    dim_word = 256
    dim_ctx = 512
    dim = 256
    alpha_c = 0.01 # alpha의 합이 1이 되도록 regularization
    decay_c = 0.001# l2 regularization


    dictionary = pd.read_pickle('/home/taeksoo/Study/Multimodal/dataset/flickr30/dictionary.pkl')
    dictionary[0] = '#START#'
    dictionary[1] = '.'
    #n_vocab = len(dictionary)

    main_model = Main_model(n_vocab, dim_word, dim_ctx, dim)

    x, mask, ctx, alphas, alpha_samples, cost = main_model.build_model()
    f_init, f_next = main_model.build_sampling_function()

    cost = cost.mean()
    updates = RMSprop(cost=cost, params=main_model.params)
    train_function = theano.function(inputs=[x,mask,ctx],
                                     outputs=cost,
                                     updates=updates,
                                     allow_input_downcast=True)

    if alpha_c > 0.:
        alpha_c = theano.shared(np.float32(alpha_c))
        alpha_reg = alpha_c * ((1. - alphas.sum(axis=0))**2).sum(axis=0).mean()
        cost += alpha_reg

    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c))
        weight_decay = 0.
        for p in main_model.params:
            weight_decay += (p ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    with open('/home/taeksoo/Study/Multimodal/dataset/flickr30/flicker_30k_align.train.pkl') as f:
        sentences = cPickle.load(f)
        image_feats = cPickle.load(f)

    sentences, image_ids = zip(*sentences)

    for start, end in zip(range(0, len(sentences)+10), range(10, len(sentences)+10, 10)):
        current_sents = sentences[start:end]
        current_sent_ind = map(lambda sent: map(lambda word: dictionary[word] if word in dictionary else None, sent.lower().split(' ')), current_sents)
        current_sent_ind = map(lambda sent: filter(lambda word: word is not None, sent), current_sent_ind)
        current_img_id = image_ids[start:end]

        current_feats = image_feats[np.array(current_img_id)]
        feat_train = np.array(current_feats.todense()).reshape(-1,196,512)

        X_train = sequence.pad_sequences(current_sent_ind, padding='post')
        mask_train = np.ones_like(X_train) * (1 - np.equal(X_train, 0))

        cost = train_function(X_train, mask_train,feat_train)

