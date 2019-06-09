# -*- coding: utf-8 -*-
#
#
#
# created by daneelwang at 2019-06-09 9:27:28
#


import time
from loguru import logger
import distance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss
import pickle
import sys

sys.path.append('../../')

from src.tools.helper import text_to_wordlist
from src.features.string_distance import distance_vec

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.linalg import norm


# 单个句子的tf
def tf_similarity(s1, s2):
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split(' '))
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


# 单个句子的tfidf
def tfidf_similarity(s1, s2):
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split(' '))
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # print(vectors)
    # 计算TFIDF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


# 共有词个数
def get_common_words(s1, s2):
    if isinstance(s1, str):
        s1 = s1.split(' ')
        s2 = s2.split(' ')
    return set(s1) & set(s2)


# 读数据
logger.info('Read data...')
train = pd.read_csv('../../data/tmp/train_sample.csv')
test_press = pd.read_csv('../../data/test/clear_press_test.csv')
test_paper = pd.read_csv('../../data/test/clear_paper_test.csv')


# 预处理
logger.info('Process train data...')
train['press_headline'] =  train.press_headline.map(lambda x:text_to_wordlist(str(x)))
train['paper_title'] = train.paper_title.map(lambda x:text_to_wordlist(str(x)))
train['press_text'] = train.press_text.map(lambda x:text_to_wordlist(str(x)))
logger.success('Process train data...')


logger.info('Process test data...')
test_press['press_headline'] = test_press.press_headline.map(lambda x:text_to_wordlist(str(x)))
test_press['press_text'] = test_press.press_text.map(lambda x:text_to_wordlist(str(x)))
test_paper['paper_title'] = test_paper.paper_title.map(lambda x:text_to_wordlist(str(x)))
logger.success('Process test data...')


logger.info('Make feature vector...')

# train['f_distance'] = train.apply(lambda r: distance_vec(r['paper_title'], r['press_headline']), axis=1)

# 编辑距离
train['f_edit_distance'] = train.apply(lambda r: distance.quick_levenshtein(r['paper_title'], r['press_headline']), axis=1)
logger.success('Make feature vector: f_edit_distance')

# jaccard比
train['f_jaccard_distance'] = train.apply(lambda r: distance.jaccard(r['paper_title'], r['press_headline']), axis=1)
logger.success('Make feature vector: f_jaccard_distance')

# sorensen距离
train['f_sorensen_distance'] = train.apply(lambda r: distance.sorensen(r['paper_title'], r['press_headline']), axis=1)
logger.success('Make feature vector: f_sorensen_distance')

# fast compare
train['f_fc_distance'] = train.apply(lambda r: distance.fast_comp(r['paper_title'], r['press_headline'], transpositions=True), axis=1)
logger.success('Make feature vector: f_fc_distance')

# 最长公共子串长度
train['f_substring_distince'] = train.apply(lambda r: distance.lcsubstrings(r['paper_title'], r['press_headline'], positions=True)[0], axis=1)
logger.success('Make feature vector: f_substring_distince')

# 公有词个数
train['f_common_words'] = train.apply(lambda r: len(get_common_words(r['paper_title'], r['press_headline'])), axis=1)
logger.success('Make feature vector: f_common_words')

logger.success('Make feature vector...')


logger.info('Make train matrix...')
col = [c for c in train.columns if c.startswith('f_')]

pos_train = train[train['label'] == 1]
neg_train = train[train['label'] == 0]
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['label'], test_size=0.2, random_state=0)

params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.02
params["subsample"] = 0.7
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 4
params["silent"] = 1
params["seed"] = 1632


logger.info('Train model...')
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

logger.info('Train model...')
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100) #change to higher #s
logger.success('Train finish')


print(log_loss(train.label, bst.predict(xgb.DMatrix(train[col]))))

with open('../../data/tmp/xgb_baseline_headline_title_1.o', 'wb') as fp:
    pickle.dump(bst, fp)








