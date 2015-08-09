from taeksoo.cnn_util import *
import pandas as pd
import numpy as np
import os
import scipy
import ipdb
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer

annotation_path = '/home/taeksoo/Study/Multimodal/dataset/flickr30/results_20130124.token'
vgg_deploy_path = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
vgg_model_path  = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_16_layers.caffemodel'
flickr_image_path = '/home/taeksoo/Study/Multimodal/dataset/flickr30/flickr30k-images'
feat_path='/home/taeksoo/Study/Multimodal/Show_Attend_Tell/feat/flickr30k'
cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

captions = annotations['caption'].values

vectorizer = CountVectorizer().fit(captions)
dictionary = vectorizer.vocabulary_
dictionary_series = pd.Series(dictionary.values(), index=dictionary.keys()) + 2
dictionary = dictionary_series.to_dict()

with open('/home/taeksoo/Study/Multimodal/dataset/flickr30/dictionary.pkl', 'wb') as f:
    cPickle.dump(dictionary, f)

images = pd.Series(annotations['image'].unique())
image_id_dict = pd.Series(np.array(images.index), index=images)

caption_image_id = annotations['image'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

for start, end in zip(range(0, len(images)+10000, 10000), range(10000, len(images)+10000, 10000)):
    image_files = images[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    if start == 0:
        feat_flatten_list = scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))
    else:
        feat_flatten_list = scipy.sparse.vstack([feat_flatten_list, scipy.sparse.csr_matrix(np.array(map(lambda x: x.flatten(), feat)))])

with open('/home/taeksoo/Study/Multimodal/dataset/flickr30/flicker_30k_alighn.train.pkl', 'wb') as f:
    cPickle.dump(cap, f)
    cPickle.dump(feat_flatten_list, f)
