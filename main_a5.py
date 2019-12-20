"""
UserWarning: The twython library has not been installed. 
Some functionality from the twitter package will not be available.
This warning is due to various version of Python.
"""

import nltk
import os
import sys
import time
import joblib

import numpy as np
import nltk.sentiment

from nltk.text import Text
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import movie_reviews
#from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

def mpqa_decode(path):
    
    """
    decode MPQA Subjectivity lexicon data file as a dict
    input: the path of dataset
    output: apqa dict {word:(neg or pos)}
    """
    mpqa = {}
    fr = open(path,"r")  
    lines = fr.readlines()
    for line in lines:
        mpqa[line.split()[2].split('=')[1]] = line.split()[-1].split('=')[1][:3]
        
    fr.close()
    
    return mpqa


class Vocabulary():
    
    """
    class to store the information of our dataset's vocabulary
    the vocabulary is build from our dataset without stopwords
    also used to findout the pos of word
    """
    
    def __init__(self, tokens_list):

        # building a reference dict for finding pos of word
        self.words_reference = {}
        # build a vocabulary set
        self.words = set()
        
        for index, tokens in enumerate(tokens_list):
            for token in tokens.words_list:
                self.words.add(token)
        
        # change set to list, used to count the pos of words
        self.words = list(self.words)
        
        for index, word in enumerate(self.words):
            self.words_reference[word] = index

        self.size = len(self.words_reference.keys())
                        
    def __str__(self):
        return f'<Vocabulary size=this voc contain {self.size} words>'

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.words[index]
    
    def pos(self, word):
        return self.words_reference[word]
    
    def reset_vocabulary_by_stopwordlist(self):
        
        new_words = set()
        self.words_reference = {}
        
        for word in self.words:
            if word not in STOPLIST:
                new_words.add(word)
                
        new_words = list(new_words)
        
        for index, word in enumerate(new_words):
            self.words_reference[word] = index
            
        self.words = new_words
        self.size = len(self.words_reference.keys())
    
    def reset_vocabulary_by_sentiwordnet(self):
        """
        reset vocabulary by sentiwordnet
        """
        self.words_reference = {}
        new_words = set()
        
        for word in self.words:
            senti_score = list(swn.senti_synsets(word, 'a'))
            # reset our vocabulary by words' senti score
            # check word is in senti dataset first
            if senti_score and (senti_score[0].pos_score()>0.5 or senti_score[0].neg_score()>0.5):
                new_words.add(word)
                
        new_words = list(new_words)
        
        for index, word in enumerate(new_words):
            self.words_reference[word] = index

        self.words = new_words
        self.size = len(self.words_reference.keys())
    
    def sentiwordnet_encode(self):
        """
        using sentiwordnet pos and neg score to encode vocabulary
        return the encode array
        """

        # we only consider the highest score and donesn't care about its pos or neg

        words_encode = np.zeros([len(self.words)], dtype = "float32")
        for index, word in enumerate(self.words):
            senti_score = list(swn.senti_synsets(word, 'a'))
            if senti_score:
                words_encode[index] = max(senti_score[0].pos_score(), senti_score[0].neg_score())
                
        return words_encode
    
    
    def reset_vocabulary_by_english_vocabulary(self):
        
        new_words = set()
        self.words_reference = {}
        
        for word in self.words:
            if word in ENGLISH_VOCABULARY:
                new_words.add(word)
                
        new_words = list(new_words)
        
        for index, word in enumerate(new_words):
            self.words_reference[word] = index
            
        self.words = new_words
        self.size = len(self.words_reference.keys())
        
    def reset_vocabulary_by_MPQA(self):

        new_words = set()
        self.words_reference = {}
        
        for word in self.words:
            if word in MPQA_VOCABULARY:
                new_words.add(word)
                
        new_words = list(new_words)
        
        for index, word in enumerate(new_words):
            self.words_reference[word] = index
            
        self.words = new_words
        self.size = len(self.words_reference.keys())

class Tokens():
    """
    class to store the tokens from text file
    take file path as input
    """
    def __init__(self, path, readed = False):

        if not readed:
            with open(path) as fr:
                self.words_list = fr.read().split()

        else:
            self.words_list = path.split()
        
        self.size = len(self.words_list)
        self.name = path.split('/')[-1]
            
    def __str__(self):
        return f"<Tokens size={self.size} name={self.name}>"
        
    def __len__(self):
        return self.size
    
    def negation_mark(self):
        #reset tokens_list with negation
        #like --> like_NEG
        self.words_list = nltk.sentiment.util.mark_negation(self.words_list)
    
    def frequence(self, vocabulary):
        # build a np array for frequence features
        # note: int8 is not enough for this dataset, when vaule more than 127 it will cause neg value
        frequence_array = np.zeros([len(vocabulary)], dtype='int16')
        
        join_set = set(self.words_list) & set(vocabulary.words)
    
    
        for token in self.words_list:
            # only count the words in vocabulary
            if token in join_set:
                index = vocabulary.pos(token)
                frequence_array[index] += 1
                
        return frequence_array
    
    def binary(self, vocabulary):
        # build a np array for binary features
        binary_array = np.zeros([len(vocabulary)], dtype='int16')
        
        join_set = set(self.words_list) & set(vocabulary.words)
        
        for token in join_set:
            index = vocabulary.pos(token)
            binary_array[index] = 1
                
        return binary_array
    
    def most_frequent_token(self, vocabulary):
        # return the first most frequent words in vocabulary
        # return the frequence of the most frequent word
        frequence_array = self.frequence(vocabulary)
        
        most_frequent_index = np.argmax(frequence_array)
        token_frequence = np.max(frequence_array)
        
        return vocabulary.words[most_frequent_index], token_frequence

def dataset_builder(tokens_list, vocabulary, frequence = True):
    
    dataset_array = np.zeros([2000, len(vocabulary)], dtype='int16')

    if frequence:
        for index, tokens in enumerate(tokens_list):
            dataset_array[index] = tokens.frequence(vocabulary)
    else:
        for index, tokens in enumerate(tokens_list):
            dataset_array[index] = tokens.binary(vocabulary)
        
    return dataset_array

def train_test_split(dataset_array, labels, test_size=0.2):
    
    rand_list = list(range(2000))
    np.random.shuffle(rand_list)
    
    split_number = int(test_size*2000)

    test_X = dataset_array[rand_list[:split_number]]
    test_Y = labels[rand_list[:split_number]]

    train_X = dataset_array[rand_list[split_number:]]
    train_Y = labels[rand_list[split_number:]]
    
    return train_X, train_Y, test_X, test_Y

def model_save_info_print(model_type, train_auc, test_auc, file_path, model, elapsed):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    joblib.dump(model, file_path)
    
    print(f'Creating {model_type} in {file_path}')
    print(f'    Elapsed time: {elapsed}s')
    print(f'    TRAIN Accuracy: {train_auc}')
    print(f'    TEST Accuracy: {test_auc}')
    print('#########################################################')
    print('#########################################################')

def nb_model(train_X, train_Y, test_X, test_Y):
    '''
    train and test a nb model
    return the model, running time and train test auc
    '''
    start = time.clock()
    model = MultinomialNB()
    model.fit(train_X, train_Y)
    y_train_pred = model.predict(train_X)
    y_test_pred = model.predict(test_X)

    train_auc = (train_Y == y_train_pred).sum()/train_X.shape[0]
    test_auc = (test_Y == y_test_pred).sum()/test_X.shape[0]

    elapsed = (time.clock() - start)

    return model, train_auc, test_auc, elapsed

def tree_model(train_X, train_Y, test_X, test_Y):

    start = time.clock()
    model = DecisionTreeClassifier(random_state=0,  max_depth=8)

    model.fit(train_X, train_Y)
    y_train_pred = model.predict(train_X)
    y_test_pred = model.predict(test_X)

    train_auc = (train_Y == y_train_pred).sum()/train_X.shape[0]
    test_auc = (test_Y == y_test_pred).sum()/test_X.shape[0]

    elapsed = (time.clock() - start)

    return model, train_auc, test_auc, elapsed

def menu():
    print('Choose a model')
    print('1 - all words raw counts')
    print('2 - all words binary')
    print('3 - SentiWordNet words')
    print('4 - Subjectivity Lexicon words')
    print('5 - all words plus Negation')

# NLTK stoplist with 3136 words
STOPLIST = set(nltk.corpus.stopwords.words())

# Vocabulary with 234,377 English words from NLTK
ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())

# about 8000 words associated with parts of speech and a subjectivity score.
MPQA_VOCABULARY = mpqa_decode('data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')



def main():
    """
    the main func
    """
    length = len(sys.argv)

    # Using the raw data, because I implemented all basic functions myself
    # it is better for me to use raw data
    documents = [(movie_reviews.raw(fileid), category) 
                    for category in movie_reviews.categories() 
                    for fileid in movie_reviews.fileids(category)]

    if length == 2:

        # data list
        tokens_list = []
        # the label list of data
        labels = []

        for document in documents:
            tokens = Tokens(document[0], True)
            tokens_list.append(tokens)
            labels.append(document[1])

## The first NB model
## The most basic one with all words
        print('The first NB model with all words and words frequence')
        # build the basic vocabulary set for our data
        vocabulary = Vocabulary(tokens_list)
        # get features from tokens_list
        dataset_array = dataset_builder(tokens_list, vocabulary, True)
        labels = np.array(labels)

        train_X, train_Y, test_X, test_Y = train_test_split(dataset_array, labels, test_size=0.1)     
        model, train_auc, test_auc, elapsed = nb_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Bayes classifier', train_auc, test_auc, 'classifiers/bayes-all-words.jbl', model, elapsed)

## The first tree model
        print('The first Tree model with all words and words frequence')
        print('Using the same training set and testing set for the tree model')
        model, train_auc, test_auc, elapsed = tree_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Decision tree classifier', train_auc, test_auc, 'classifiers/decision-tree-all-words.jbl', model, elapsed)

## The second NB model
## words only from sentiwordnet which pos and neg score more than 0.5
        print('The second NB model with words in sentiwordnet and binary features')
        print("update vocabulary dataset")
        vocabulary.reset_vocabulary_by_sentiwordnet()

        dataset_array = dataset_builder(tokens_list, vocabulary, False)
        train_X, train_Y, test_X, test_Y = train_test_split(dataset_array, labels, test_size=0.1)
        model, train_auc, test_auc, elapsed = nb_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Bayes classifier', train_auc, test_auc, 'classifiers/bayes-sentiwordnet-words.jbl', model, elapsed)

## The second tree model
        print('The second Tree model with words in sentiwordnet and words frequence')
        model, train_auc, test_auc, elapsed = tree_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Decision tree classifier', train_auc, test_auc, 'classifiers/decision-tree-sentiwordnet-words.jbl', model, elapsed)

## The third NB model
## All words with binary features
        print('The third NB model with all words and binary features')
        # build the basic vocabulary set for our data
        vocabulary = Vocabulary(tokens_list)
        dataset_array = dataset_builder(tokens_list, vocabulary, False)
        train_X, train_Y, test_X, test_Y = train_test_split(dataset_array, labels, test_size=0.1)
        model, train_auc, test_auc, elapsed = nb_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Bayes classifier', train_auc, test_auc, 'classifiers/bayes-all-words-binary.jbl', model, elapsed)

## The third tree model
        print('The third Tree model with all words and binary features')
        model, train_auc, test_auc, elapsed = tree_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Decision tree classifier', train_auc, test_auc, 'classifiers/decision-tree-all-words-binary.jbl', model, elapsed)

## The fourth NB model
## Words from MPQA dataset
        print('The fourth NB model with MPQA dataset and binary features')
        vocabulary.reset_vocabulary_by_MPQA()
        dataset_array = dataset_builder(tokens_list, vocabulary, False)
        train_X, train_Y, test_X, test_Y = train_test_split(dataset_array, labels, test_size=0.1)
        model, train_auc, test_auc, elapsed = nb_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Bayes classifier', train_auc, test_auc, 'classifiers/bayes-MPQA.jbl', model, elapsed)

## The fourth tree model
        print('The fourth Tree model with MPQA dataset and binary features')
        model, train_auc, test_auc, elapsed = tree_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Decision tree classifier', train_auc, test_auc, 'classifiers/decision-tree-MPQA.jbl', model, elapsed)

## The fiveth NB model
## All words with negation 'don't like it. -> don't like_NEG it_NEG'
        print('The fiveth model, All words with negation')
        print('rebuild an dataset which consider negation words')

        # data list
        tokens_list = []
        # the label list of data
        labels = []

        for document in documents:
            tokens = Tokens(document[0], True)
            tokens.negation_mark()
            tokens_list.append(tokens)
            labels.append(document[1])

        vocabulary = Vocabulary(tokens_list)
        dataset_array = dataset_builder(tokens_list, vocabulary, False)
        labels = np.array(labels)
        train_X, train_Y, test_X, test_Y = train_test_split(dataset_array, labels, test_size=0.1)
        model, train_auc, test_auc, elapsed = nb_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Bayes classifier', train_auc, test_auc, 'classifiers/bayes-all-words-negition-binary.jbl', model, elapsed)

## The fiveth tree model
        print('The fiveth Tree model with all words with negation')
        model, train_auc, test_auc, elapsed = tree_model(train_X, train_Y, test_X, test_Y)

        # saving model and print result
        model_save_info_print('Decision tree classifier', train_auc, test_auc, 'classifiers/decision-tree-all-words-negition-binary.jbl', model, elapsed)

    elif length == 4:
        model_name = sys.argv[2]
        filename = sys.argv[3]

        menu()
        number = input("Type a number:")

        # data list
        tokens_list = []

        for document in documents:
            tokens = Tokens(document[0], True)

            if number == '5':
                tokens.negation_mark()

            tokens_list.append(tokens)

        vocabulary = Vocabulary(tokens_list)

        if model_name == 'bayes' or model_name == 'tree':
            if number == '1':
                dataset_array = np.zeros([1, len(vocabulary)], dtype='int16')
                if model_name == 'bayes':
                    model = joblib.load('classifiers/bayes-all-words.jbl')
                if model_name == 'tree':
                    model = joblib.load('classifiers/decision-tree-all-words.jbl')
                tokens = Tokens(filename, False)
                token_array = tokens.frequence(vocabulary)
                dataset_array[0] = token_array
                print(model.predict(dataset_array)[0])

            elif number == '2':
                dataset_array = np.zeros([1, len(vocabulary)], dtype='int16')
                if model_name == 'bayes':
                    model = joblib.load('classifiers/bayes-all-words-binary.jbl')
                if model_name == 'tree':
                    model = joblib.load('classifiers/decision-tree-all-words-binary.jbl')
                tokens = Tokens(filename, False)
                token_array = tokens.binary(vocabulary)
                dataset_array[0] = token_array
                print(model.predict(dataset_array)[0])

            elif number == '3':
                vocabulary.reset_vocabulary_by_sentiwordnet()
                dataset_array = np.zeros([1, len(vocabulary)], dtype='int16')
                if model_name == 'bayes':
                    model = joblib.load('classifiers/bayes-sentiwordnet-words.jbl')
                if model_name == 'tree':
                    model = joblib.load('classifiers/decision-tree-sentiwordnet-words.jbl')
                tokens = Tokens(filename, False)
                token_array = tokens.binary(vocabulary)
                dataset_array[0] = token_array
                print(model.predict(dataset_array)[0])

            elif number == '4':
                vocabulary.reset_vocabulary_by_MPQA()
                dataset_array = np.zeros([1, len(vocabulary)], dtype='int16')
                if model_name == 'bayes':
                    model = joblib.load('classifiers/bayes-MPQA.jbl')
                if model_name == 'tree':
                    model = joblib.load('classifiers/decision-tree-MPQA.jbl')
                tokens = Tokens(filename, False)
                token_array = tokens.binary(vocabulary)
                dataset_array[0] = token_array
                print(model.predict(dataset_array)[0])

            elif number == '5':
                dataset_array = np.zeros([1, len(vocabulary)], dtype='int16')
                if model_name == 'bayes':
                    model = joblib.load('classifiers/bayes-all-words-negition-binary.jbl')
                if model_name == 'tree':
                    model = joblib.load('classifiers/decision-tree-all-words-negition-binary.jbl')
                tokens = Tokens(filename, False)
                tokens.negation_mark()
                token_array = tokens.binary(vocabulary)
                dataset_array[0] = token_array
                print(model.predict(dataset_array)[0])

            else:
                print("number out of range")
        else:
            print('Please select bayes or tree')

    else:
        print("Running example: ")
        print("for training : python3 main_a5.py --train ")
        print("for text testing : python3 main_a5.py --run 'bayes'|'tree' <filename>")

if __name__ == "__main__":
    main()