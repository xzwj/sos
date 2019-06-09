# -*- coding: utf-8 -*-
#
#
#
# created by wangquan at 2019-06-09 3:7:46
#


import pandas as pd
from string import punctuation
import re
import nltk


# 读所有语句，拼接成一个
def read_all_sentence(data_types=['train', 'test']):
    all_sentence_list = []

    for dt in data_types:
        base_dir = '../../data/{}/'
        base_dir = base_dir.format(dt)

        press_path = base_dir + 'clear_press_{}.csv'.format(dt)
        paper_path = base_dir + 'paper_{}.csv'.format(dt)
        df_press = pd.read_csv(press_path)
        df_paper = pd.read_csv(paper_path)

        sentences = df_press['press_headline'].tolist() + df_press['press_text'].tolist() +  df_paper['paper_title'].tolist()

        all_sentence_list += sentences

    result = '\n'.join(str(i) for i in all_sentence_list)
    return result


stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then',
              'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to',
              'What', 'Which',
              'Is', 'If', 'While', 'This']


def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = text.rstrip('?')
    text = text.rstrip(',')
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        # stemmer = SnowballStemmer('english')
        # stemmed_words = [stemmer.stem(word) for word in text]
        stemmed_words = [nltk.PorterStemmer().stem_word(word.lower()) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


# 构造训练样本对
def make_train_sample():
    base_dir = '../../data/train/'
    dt = 'train'
    press_path = base_dir + 'clear_press_{}.csv'.format(dt)
    paper_path = base_dir + 'paper_{}.csv'.format(dt)
    match_path = base_dir + 'match.csv'

    df_press = pd.read_csv(press_path)
    df_paper = pd.read_csv(paper_path)
    df_match = pd.read_csv(match_path)

    df_sample_T = df_match.merge(df_press, how='inner', on='press_id')
    df_sample_T = df_sample_T.merge(df_paper, how='inner', on='paper_id')
    df_sample_T['label'] = 1

    df_match_F = df_match.copy()
    df_match_F['paper_id'] = df_match_F.paper_id.tolist()[::-1]
    df_sample_F = df_match_F.merge(df_press, how='inner', on='press_id')
    df_sample_F = df_sample_F.merge(df_paper, how='inner', on='paper_id')
    df_sample_F['label'] = 0

    df_sample = pd.concat([df_sample_T, df_sample_F])
    df_result = df_sample.sample(frac=1).reset_index(drop=True)

    return df_result


def make_predict_sample():
    pass



if __name__ == '__main__':
    # df = make_train_sample()
    # print(df.head())
    # print(df.describe())
    # df.to_csv('../../data/tmp/train_sample.csv', index=False)

    all_sentence = read_all_sentence()
    clear_sentence = text_to_wordlist(all_sentence)
    with open('../../data/tmp/all_sentence.txt', 'w+') as fp:
        fp.write(clear_sentence)












