#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
import numpy as np
import nltk
import datetime
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.optimize import minimize
stops = set(stopwords.words("english"))
import xgboost as xgb
from sklearn.model_selection import train_test_split
import multiprocessing
import difflib
import pickle
import re
from string import punctuation
import sys

from loguru import logger

from gensim.models import Word2Vec
from nltk.corpus import brown 
from gensim.models.keyedvectors import KeyedVectors

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
              'Is','If','While','This']


def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = text.rstrip('?')
    text = text.rstrip(',')

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        #stemmer = SnowballStemmer('english')
        #stemmed_words = [stemmer.stem(word) for word in text]
        stemmed_words = [nltk.PorterStemmer().stem_word(word.lower()) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


#word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
br = Word2Vec(brown.sents())


def get_similar(word):
    if word in br:
        lis = br.most_similar(word, topn=3)
        ret = []
        for one in lis:
            ret.append(one[0])
        return ret
    else:
        return [word]


logger.info('Read data...')
train = pd.read_csv('../../data/tmp/train_sample.csv')
test_press = pd.read_csv('../../data/test/clear_press_test.csv')
test_paper = pd.read_csv('../../data/test/clear_paper_test.csv')


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


train_qs = pd.Series(train['press_headline'].tolist() + train['paper_title'].tolist() + train['press_text'].tolist() + test_press['press_headline'].tolist() + test_press['press_text'].tolist() + test_paper['paper_title'].tolist()).astype(str)


tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
#cvect = CountVectorizer(stop_words='english', ngram_range=(1, 1))


logger.info('Train tfidf...')
tfidf_txt = train_qs.copy()
tfidf.fit_transform(tfidf_txt)
#cvect.fit_transform(tfidf_txt)
logger.success('Train tfidf...')

pickle.dump(tfidf, '../../data/tmp/tfidf_123gram.o')


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()


def word_match_share(row, c1, c2):
    q1words = {}
    q2words = {}
    for word in str(row[c1]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[c2]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def tfidf_word_match_share(q1words, q2words):
    word_1 = []
    for word in q1words:
        word_1.extend(get_similar(word))

    word_2 = []
    for word in q2words:
        word_2.extend(get_similar(word))

    shared_weights = [0] + [weights.get(w, 0) for w in word_1 if w in word_2] + [weights.get(w, 0) for w in word_2 if w in word_1]
    total_weights = [weights.get(w, 0) for w in word_1] + [weights.get(w, 0) for w in word_2]

    if (np.sum(shared_weights) == 0):
        return 0

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def get_features(df_features, c1, c2):
    logger.info('Get features...')
    # now = datetime.datetime.now()
    # print now.strftime('%Y-%m-%d %H:%M:%S') 
    # print "matchnouns"
    # df_features['question1_nouns'] = df_features[c1].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # df_features['question2_nouns'] = df_features[c2].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # #df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
    # df_features['z_noun_match'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_nouns, r.question2_nouns), axis = 1)
    
    # now = datetime.datetime.now()
    # print now.strftime('%Y-%m-%d %H:%M:%S')   
    # print "matchverb"
    # df_features['question1_verbs'] = df_features[c1].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[0] == 'V' and t[1] == 'B'])
    # df_features['question2_verbs'] = df_features[c2].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[0] == 'V' and t[1] == 'B'])
    # #df_features['z_verb_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_verbs if w in r.question2_verbs]), axis=1)  #takes long
    # df_features['z_verb_match'] = df_features.apply(lambda r : tfidf_word_match_share(r.question1_verbs, r.question2_verbs), axis = 1)
    
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print("stem_tfidf")
    df_features['q1_stem'] = df_features[c1].map(lambda x: [w for w in nltk.PorterStemmer().stem_word(str(x).lower()).split(' ')])
    df_features['q2_stem'] = df_features[c2].map(lambda x: [w for w in nltk.PorterStemmer().stem_word(str(x).lower()).split(' ')])
    #df_features['z_adj_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_adjs if w in r.question2_adjs]), axis=1)  #takes long
    df_features['z_stem_tfidf'] = df_features.apply(lambda r : tfidf_word_match_share(r.q1_stem, r.q2_stem), axis = 1)
    now = datetime.datetime.now()
    # print now.strftime('%Y-%m-%d %H:%M:%S')
    # print('w2v tfidf...')
    # df_features['z_tfidf_w2v'] = df_features.apply(lambda r : tfidf_word_match_share(r[c1].tolist(), r[c2].tolist()), axis = 1)
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('nouns...')
    df_features['question1_nouns'] = df_features[c1].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df_features['question2_nouns'] = df_features[c2].map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
    print('lengths...')
    df_features['z_len1'] = df_features[c1].map(lambda x: len(str(x)))
    df_features['z_len2'] = df_features[c2].map(lambda x: len(str(x)))
    df_features['z_word_len1'] = df_features[c1].map(lambda x: len(str(x).split()))
    df_features['z_word_len2'] = df_features[c2].map(lambda x: len(str(x).split()))
    now = datetime.datetime.now()
    print(now.strftime('%Y-%m-%d %H:%M:%S'))
    print('difflib...')
    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
    print('word match...')
    df_features['z_word_match'] = df_features.apply(lambda x: word_match_share(x, c1, c2), axis=1, raw=True)
    print('tfidf...')
    df_features['z_tfidf_sum1'] = df_features[c1].map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_sum2'] = df_features[c2].map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean1'] = df_features[c1].map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean2'] = df_features[c2].map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len1'] = df_features[c1].map(lambda x: len(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len2'] = df_features[c2].map(lambda x: len(tfidf.transform([str(x)]).data))
    return df_features.fillna(0.0)


train = get_features(train, 'press_headline', 'paper_title')
train.to_csv('../../data/tmp/train_feature_headline_and_title.csv', index=False)
logger.success('Save features...')


col = [c for c in train.columns if c[:1]=='z']

pos_train = train[train['label'] == 1]
neg_train = train[train['label'] == 0]
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.2, random_state=0)

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
bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100) #change to higher #s
logger.success('Train finish')
print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))

pickle.dump(bst, '../../data/tmp/xgb_baseline_headline_title.o')

# test = get_features(test)
# test.to_csv('test_feature_clean.csv', index=False)

# sub = pd.DataFrame()
# sub['test_id'] = test['test_id']
# sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))
#
# sub.to_csv('submission_xgb_w2v_clean_02_04.csv', index=False)
