"""main_3a.py
An instance of the Text class should be initialized with a file path (a file or
directory). The example here uses object as the super class, but you may use
nltk.text.Text as the super class.
An instance of the Vocabulary class should be initialized with an instance of
Text (not nltk.text.Text).
"""
import os
import nltk
import re
from nltk.corpus import brown, words
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.text import Text as tx
from nltk.collocations import BigramCollocationFinder

class FSA(object):
    def __init__(self, test, states, final_states, transitions):
        self.test = test
        self.states = states
        self.final_states = final_states
        self.transitions = transitions
        self.symbol_to_states = {symbol: (start, end) for start, symbol, end in self.transitions}

    def pp(self):
        dict_pp = {state: {} for state in self.states}
        for start_state, symbol, end_state in self.transitions:
            dict_pp[start_state][symbol] = end_state

        for state in dict_pp.keys():
            if dict_pp[state] == {}:
                print(f"<State {state} f>")
            else:
                print(f"<State {state}>")
                for symbol in dict_pp[state]:
                    print(f"    {symbol} --> {dict_pp[state][symbol]}")

    def accept(self, target):
        if ' ' not in target:
            symbols = list(target)
        else:
            symbols = target.spilt(' ')

        trans = [self.symbol_to_states[symbol] for symbol in symbols]

        if len(symbols) == 1:
            return symbols[0] in self.symbol_to_states
        else:
            output = [trans[i+1][0] == trans[i][1] for i in range(len(trans)-1)]
            return all(output)


class Text(object):

    def __init__(self, path):
        self.path = path
        if os.path.isfile(self.path):
            file = open(self.path, 'r')
            self.raw = file.read()
            self.text = tx(word_tokenize(self.raw))
        elif os.path.isdir(self.path):
            files = '.*\.*'
            corpu = PlaintextCorpusReader(path, files)
            self.raw = corpu.raw()
            self.text = tx(corpu.words())
    
    def __len__(self):
        return len(self.text)
    
    def token_count(self):
        return len(self.text)
        
    def type_count(self):
        return len(set(self.text))
    
    def sentence_count(self):
        text = tx(sent_tokenize(self.raw))
        return len(text)
    
    def most_frequent_content_words(self):
        stop = stopwords.words('english')
        new_text = []
        for word in self.text:
            if word.lower() not in stop:
                if word.isalpha():
                    new_text.append(word)
                elif any([i.isalpha() for i in list(word)]):
                    if word[0].isalpha():
                        new_text.append(word)
        return FreqDist(new_text).most_common(25)
    
    def most_frequent_bigrams(self):
        stop = stopwords.words('english')
        new_text = []
        for word in self.text:
            if word.lower() not in stop:
                if word.isalpha():
                    new_text.append(word)
                elif any([i.isalpha() for i in list(word)]):
                    if word[0].isalpha():
                        new_text.append(word)          
        finder = BigramCollocationFinder.from_words(new_text)
        freq = finder.ngram_fd
        return freq.most_common(25)
    
    def find_sirs(self):
        used_string = self.raw
        out = re.findall(r'Sir\s[A-Z][\w+\-]*', used_string)
        return sorted(list(set(out)))
    
    def find_brackets(self):
        used_string = self.raw
        out = re.findall(r'\([\.]*[\w\s]+[\.|\!]*\)', used_string)
        final = [i[1] for i in out]
        pente = sorted(list(set(final)))
        out = re.findall(r'([\[].+?[\]])', used_string)
        square = sorted(list(set(out)))
        return pente + square

    def find_roles(self):
        used_string = self.raw
        out = re.findall(r'(([A-Z]{3,}.{1,30})\:)', used_string)
        final = [i[1] for i in out]
        return sorted(list(set(final)))

    def find_repeated_words(self):
        used_string = self.raw
        out = re.findall(r'((\w{3,})(\s\2){2})', used_string)
        final = [i[0] for i in out]
        return sorted(list(set(final)))

    def apply_fsa(self, fsa):
        start_state = [item[1] for item in fsa.transitions if 'S0' in item]
        index_sir = [(i, state) for i in range(len(self.text.tokens)) for state in start_state if self.text.tokens[i] == state]
        output = [(i, state + ' ' + self.text.tokens[i+1]) for i, state in index_sir if self.text.tokens[i+1] in fsa.symbol_to_states]
        return output
        
    




class Vocabulary(object):
    def __init__(self, text):
        self.text = text.text
        
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

