# -*- encoding: utf-8 -*-
#
#
#
# created by wangquan at 2019-06-08 20:19:06
#

import pandas as pd
import numpy as np
import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from loguru import logger
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


# 计算各种距离
def distance_vec(s1, s2):
    edit_distance = distance.levenshtein(s1, s2)
    jaccard_distance = distance.jaccard(s1, s2)
    sorensen_distance = distance.sorensen(s1, s2)
    # hamming_distnace = distance.hamming(s1, s2)
    fc_distance = distance.fast_comp(s1, s2, transpositions=True)
    substring_distince = distance.lcsubstrings(s1, s2, positions=True)[0]
    common_words_distcance = len(get_common_words(s1, s2))
    tf_distance = tf_similarity(s1, s2)
    tfidf_distance = tfidf_similarity(s1, s2)
    vec = np.array([
        edit_distance,          # 编辑距离
        jaccard_distance,       # jaccard距离
        sorensen_distance,      # sorensen
        # hamming_distnace,     # 汉明距离
        fc_distance,            # fast commaon
        substring_distince,     # 最长公共子串长度
        common_words_distcance, # 公共词个数
        tf_distance,            # 单文本tf
        tfidf_distance          # 单文本tfidf
    ])
    return vec