"""I decided to use Decision Tree to predict the POS of words since we have learned it before."""
"""The idea is to analyze each word's features, including its position in a sentence, its suffix, """
"""its previous and next words, etc. According to last several suffix, it would easier to detect nouns"""
"""or adjective, or adverb. Usually, all capitalized, or first word capitalized within a sentence, are """
"""very likely to be nous; ending with 'ly' is likely to adjective, ect. Therefore, I will extract the"""
"""the features of each word from Tagged Sentence list from Brown, and transfer it into dataframe with"""
"""each feature as a column. Then I will fit a decision tree model by using information gain patten(entropy)"""

import nltk
from nltk.corpus import brown

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


def get_word_feature(sent, word_pos):
    """return the word features as a dictionary since it is easy to convert to dataframe"""
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
    """getting the word out of the tuple"""
    return [w for w,t in tagged_sent]

def to_dataset(tagged_sent):
    """Convert the list to dataframe as x and y. The input parameter is a list of sentence with tuple that"""
    """contains the word and tag. We can use the brown.tagged_sents() to get this parameter."""
    x = []
    y = []
    for sent in tagged_sent:
        for i in range(len(sent)):
            x.append(get_word_feature(get_words(sent), i))
            y.append(sent[i][1])
    return x, y

def get_trained_model(x_train, y_train):
    """Since each factor in x is a dictionary, I would change it as a matrix."""
    """Since the features size is large, I used sparse matrix to save the memory."""
    clf = Pipeline([
        ('vectorize', DictVectorizer(sparse = True)),
        ('classifier', DecisionTreeClassifier(criterion = 'entropy'))
    ])
    clf.fit(x_train, y_train)
    return clf

def get_pos_tag(model, word_list):
    """This function is used to tagged a single sentence. Thus, the input parmeter 'word_list' is a list of word of"""
    """a sentence, we can use the function word_tokenize from nltk to get the word list."""
    tag = model.predict([get_word_feature(word_list, i) for i in range(len(word_list))])
    return list(zip(word_list, tag))