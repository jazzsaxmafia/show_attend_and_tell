from taeksoo.cnn_util import *
import pandas as pd
import numpy as np
import os
import ipdb
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer

annotation_path = '/home/taeksoo/Study/Multimodal/dataset/coco/annotations/captions_train2014.json'
vgg_deploy_path = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
vgg_model_path  = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_16_layers.caffemodel'
coco_image_path = '/home/taeksoo/Study/Multimodal/dataset/coco/train2014'

cnn = CNN(deploy=vgg_deploy_path,
          model=vgg_model_path,
          batch_size=20,
          width=224,
          height=224)

with open(annotation_path) as f:
    data = json.load(f)

annotations = pd.DataFrame(data['annotations'])
captions = annotations['caption'].values

vectorizer = CountVectorizer().fit(captions)
dictionary = vectorizer.vocabulary_

images = pd.DataFrame(data['images'])
images['file_path'] = images['file_name'].map(lambda x: os.path.join(coco_image_path, x))
image_id_dict = pd.Series(np.array(images.index + 2), index=images['id'].values)

caption_image_id = annotations['image_id'].map(lambda x: image_id_dict[x]).values
cap = zip(captions, caption_image_id)

for start, end in zip(range(0, len(images)+10000, 10000), range(10000, len(images)+10000, 10000)):
    image_files = images['file_path'].values[start:end]
    feat = cnn.get_features(image_list=image_files, layers='conv5_3', layer_sizes=[512,14,14])
    ipdb.set_trace()

