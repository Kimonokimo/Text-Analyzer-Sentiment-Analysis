"""The idea of tagging the universal tag is very simple and similar to the one before."""
"""I only add one new function to it. This function is to chanage the original training"""
"""tagged sample to univseral tag. And then, train the model with new tag and predict the"""
"""outcome with new tags as well. In addition, the testing sample was changed as well."""

import nltk
import requests
import sys

from nltk.corpus import brown

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline



def change_to_uni(target):
    """I decided to change the original train and test tag before fitting the models"""

    link = 'https://raw.githubusercontent.com/slavpetrov/universal-pos-tags/master/en-brown.map'
    f = requests.get(link)
    no_space = f.text.split('\n')
    tag_dic = {}
    for i in no_space[:-1]:
        tag_list = i.split('\t')
        tag_dic[tag_list[0]] = tag_list[1]
    out = []
    for sent in target:
        out_sent = []
        for group in sent:
            word = group[0]
            if group[1] in tag_dic:
                uni_tag = tag_dic[group[1]]
                out_sent.append((word, uni_tag))
            else:
                uni_tag = group[1]
                out_sent.append((word, uni_tag))
        out.append(out_sent)
    return out

"""Descision Tree Method"""
def get_word_feature(sent, word_pos):
    return {
        'word': sent[word_pos],
        'first': word_pos == 0,
        'last': word_pos == len(sent) - 1,
        'capitalized': sent[word_pos][0].upper() == sent[word_pos][0],
        'all_caps': sent[word_pos].upper() == sent[word_pos],
        'all_lower': sent[word_pos].lower() == sent[word_pos],
        'first_1': sent[word_pos][0],
        'first_2': sent[word_pos][:2],
        'first_3': sent[word_pos][:3],
        'last_1': sent[word_pos][-1],
        'last_2': sent[word_pos][-2:],
        'last_3': sent[word_pos][-3:],
        'prev_word': '' if word_pos == 0 else sent[word_pos - 1],
        'next_word': '' if word_pos == len(sent) - 1 else sent[word_pos + 1],
        'has_hyphen': '-' in sent[word_pos],
        'is_numeric': sent[word_pos].isdigit(),
        'capitals_inside': sent[word_pos][1:].lower() != sent[word_pos][1:]
    }

def get_words(tagged_sent):
    return [w for w,t in tagged_sent]

def to_dataset(tagged_sent):
    x = []
    y = []
    for sent in tagged_sent:
        for i in range(len(sent)):
            x.append(get_word_feature(get_words(sent), i))
            y.append(sent[i][1])
    return x, y

def get_trained_model(x_train, y_train):
    clf = Pipeline([
        ('vectorize', DictVectorizer(sparse = True)),
        ('classifier', DecisionTreeClassifier(criterion = 'entropy'))
    ])
    clf.fit(x_train, y_train)
    return clf

def get_pos_tag(model, word_list):
    tag = model.predict([get_word_feature(word_list, i) for i in range(len(word_list))])
    return list(zip(word_list, tag))

if __name__ == "__main__":
    take_arg = sys.argv
    news = change_to_uni(brown.tagged_sents(categories = 'news'))

    if len(take_arg) == 2:
        method = take_arg[1]
        if 'train' in method:
            train_x, train_y = to_dataset(news)
            model1 = get_trained_model(train_x, train_y)
        else:
            print('Please check the input')       
    elif len(take_arg) == 3:
        train_x, train_y = to_dataset(news)
        model1 = get_trained_model(train_x, train_y)
        method = take_arg[1]
        argv = take_arg[2]
        if 'run' in method and isinstance(argv, str):
            sents = argv.split(' ')
            print(get_pos_tag(model1, sents))
        elif 'test' in method:
            data_tag = change_to_uni(brown.tagged_sents(categories = argv))
            test_x, test_y = to_dataset(data_tag)
            print(model1.score(test_x, test_y))