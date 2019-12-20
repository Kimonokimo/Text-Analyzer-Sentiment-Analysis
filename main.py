"""main.py

Code scaffolding

"""

import os
import nltk
from nltk.corpus import brown, words
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from nltk.text import Text

### Reading source data
def read_text(path):
    if os.path.isfile(path):
        file = open(path, 'r')
        raw = file.read()
        out = Text(word_tokenize(raw))
        file.close()
        return out
    elif os.path.isdir(path):
        files = '.*\.*'
        corpus = PlaintextCorpusReader(path, files)
        out  = Text(corpus.words())
        return out


def token_count(text):
    return len(text)


def type_count(text):
    return len(set(text))


def sentence_count(text):
    fdist = FreqDist(text)
    return fdist['.'] + fdist['!'] + fdist['?']


def most_frequent_content_words(text):
    stop = stopwords.words('english')
    new_text = []
    for word in text:
        if word.lower() not in stop:
            if word.isalpha():
                new_text.append(word)
            elif any([i.isalpha() for i in list(word)]):
                if word[0].isalpha():
                    new_text.append(word)
    return FreqDist(new_text).most_common(25)


def most_frequent_bigrams(text):
    stop = stopwords.words('english')
    new_text = []
    for word in text:
        if word.lower() not in stop:
            if word.isalpha():
                new_text.append(word)
            elif any([i.isalpha() for i in list(word)]):
                if word[0].isalpha():
                    new_text.append(word)
    finder = BigramCollocationFinder.from_words(new_text)
    return finder.ngram_fd.most_common(25)
    


class Vocabulary(object):

    def __init__(self, text):
        self.text = text

    def frequency(self, word):
        freq_list = FreqDist(self.text)
        try:
            return freq_list[word]
        except:
            return 0
        

    def pos(self, word):
        english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        if word.lower() in english_vocab:
            pos_name = nltk.pos_tag(nltk.word_tokenize(word))[0][1]
            if pos_name.startswith('N'):
                return 'n'
            if pos_name.startswith('V'):
                return 'v'
            if pos_name.startswith('J'):
                return 'a'
            if pos_name.startswith('R'):
                return 'r'
            else:
                return None
        else:
            return None

    def gloss(self, word):
        meaning_list = wn.synsets(word)
        if len(meaning_list) >= 1:
            return meaning_list[0].definition()
        else:
            return 'None'

    def kwic(self, word):
        return self.text.concordance(word)


categories = ('adventure', 'fiction', 'government', 'humor', 'news')


def compare_to_brown(text):
    categories = ('adventure', 'fiction', 'government', 'humor', 'news')
    new_text = []
    for cat in categories:
        cat_text = brown.words(categories = cat)
        for word in cat_text:
            if word.isalpha():
                new_text.append(word.lower())
            elif any([i.isalpha() for i in list(word.lower())]):
                if word[0].isalpha():
                    new_text.append(word)
    dim = set(new_text)
    print(len(dim))
    target_freq = FreqDist(text)
    for cat in categories:
        dot = 0
        v1_abs = 0
        v2_abs = 0
        texts = brown.words(categories = cat)
        fdist = FreqDist(texts)
        for word in dim:
            try:
                v1 = target_freq[word]
            except:
                v1 = 0
            try:
                v2 = fdist[word]
            except:
                v2 = 0
            dot += (v1*v2)
            v1_abs += (v1**2)
            v2_abs += (v2**2)
        cosine = dot/((v1_abs**(1/2))*(v2_abs**(1/2)))
        out_line = '{:<12}  {:>8}'.format(cat, "%.2f" % round(cosine, 2))
        print(out_line)




if __name__ == '__main__':

    text = read_text('data/grail.txt')
    token_count(text)
