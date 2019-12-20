import re
import os
import math
import sys
import nltk
import postag

from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.collections import Counter
from nltk.tokenize import word_tokenize


from fsa import FSA


# NLTK stoplist with 3136 words (multilingual)
STOPLIST = set(nltk.corpus.stopwords.words())

# Vocabulary with 234,377 English words from NLTK
ENGLISH_VOCABULARY = set(w.lower() for w in nltk.corpus.words.words())

# The five categories from Brown that we are using
BROWN_CATEGORIES = ('adventure', 'fiction', 'government', 'humor', 'news')

# Global place to store Brown vocabularies so you calculate them only once
BROWN_VOCABULARIES = None


def is_content_word(word):
    """A content word is not on the stoplist and its first character is a letter."""
    return word.lower() not in STOPLIST and word[0].isalpha()


class Text(object):
    
    def __init__(self, path, name=None):
        """Takes a file path, which is assumed to point to a file or a directory, 
        extracts and stores the raw text and also stores an instance of nltk.text.Text."""
        self.name = name
        if os.path.isfile(path):
            self.raw = open(path).read()
        elif os.path.isdir(path):
            corpus = PlaintextCorpusReader(path, '.*.mrg')
            self.raw = corpus.raw()
        self.text = nltk.text.Text(nltk.word_tokenize(self.raw))
        self.word_tag = nltk.pos_tag(self.text)
        self.words = word_tokenize(self.raw)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i]

    def __str__(self):
        name = '' if self.name is None else " '%s'" % self.name 
        return "<Text%s tokens=%s>" % (name, len(self))

    def token_count(self):
        """Just return the length of the text."""
        return len(self)

    def type_count(self):
        """Returns the type count, with minimal normalization by lower casing."""
        # an alternative would be to use the method nltk.text.Text.vocab()
        return len(set([w.lower() for w in self.text]))

    def sentence_count(self):
        """Return number of sentences, using the simplistic measure of counting period,
        exclamation marks and question marks."""
        # could also use nltk.sent.tokenize on self.raw
        return len([t for t in self.text if t in '.!?'])

    def most_frequent_content_words(self):
        """Return a list with the 25 most frequent content words and their
        frequencies. The list has (word, frequency) pairs and is ordered
        on the frequency."""
        dist = nltk.FreqDist([w for w in self.text if is_content_word(w.lower())])
        return dist.most_common(n=25)

    def most_frequent_bigrams(self, n=25):
        """Return a list with the 25 most frequent bigrams that only contain
        content words. The list returned should have pairs where the first
        element in the pair is the bigram and the second the frequency, as in
        ((word1, word2), frequency), these should be ordered on frequency."""
        filtered_bigrams = [b for b in list(nltk.bigrams(self.text))
                            if is_content_word(b[0]) and is_content_word(b[1])]
        dist = nltk.FreqDist([b for b in filtered_bigrams])
        return dist.most_common(n=n)

    def concordance(self, word):
        self.text.concordance(word)

    ## new methods for search part of assignment 3
    
    def search(self, pattern):
        return re.finditer(pattern, self.raw)

    def find_sirs(self):
        answer = set()
        for match in self.search(r"\bSir \S+\b"):
            answer.add(match.group())
        return sorted(answer)

    def find_brackets(self):
        answer = set()
        # use a non-greedy match on the characters between the brackets
        for match in self.search(r"([\(\[\{]).+?([\)\]\}])"):
            brackets = "%s%s" % (match.group(1), match.group(2))
            # this tests for matching pairs
            if brackets in ['[]', '{}', '()']:
                answer.add(match.group())
        return sorted(answer)

    def find_roles(self):
        answer = set()
        for match in re.finditer(r"^([A-Z]{2,}[^\:]+): ", self.raw, re.MULTILINE):
            answer.add(match.group(1))
        return sorted(answer)

    def find_repeated_words(self):
        answer = set()
        for match in self.search(r"(\w{3,}) \1 \1"):
            answer.add(match.group())
        return sorted(answer)

    def apply_fsa(self, fsa):
        i = 0
        results = []
        while i < len(self):
            match = fsa.consume(self.text[i:])
            if match:
                results.append((i, match))
                i += len(match)
            else:
                i += 1
        return results

    """Codes above are from sample answer. Function fsa is from sample answer as well"""
    """Starting with a4 part"""
    def nouns_more_common_in_plural_form(self):
        words_count = Counter(self.words)
        plural_list = [i[0] for i in self.word_tag if i[0].endswith('s') and i[1].endswith('S')]
        word_list = set(plural_list)
        out_list = []
        for words in word_list:
            single = words[:-1]
            if words_count[words] > words_count[single]:
                out_list.append(words)
        return sorted(out_list)
    
    def which_word_has_greatest_number_of_distinct_tags(self):
        word_tag_dict = {}
        for group in self.word_tag:
            word = group[0].lower()
            pos = group[1]
            if word in word_tag_dict and word[0].isalpha():
                word_tag_dict[word].add(pos)
            elif word not in word_tag_dict:
                word_tag_dict[word] = {pos}
        max_number = 0
        out_list = []
        for word in word_tag_dict:
            if len(word_tag_dict[word]) > max_number:
                max_number = len(word_tag_dict[word])
                out = (word, list(word_tag_dict[word]))
                out_list = [out]
            elif len(word_tag_dict[word]) == max_number:
                out = (word, list(word_tag_dict[word]))
                out_list.append(out)
        return out_list
    
    def tags_in_order_of_decreasing_frequency(self):
        tag_list = []
        for group in self.word_tag:
            pos = group[1]
            tag_list.append(pos)
        freq = FreqDist(tag_list)
        return freq.most_common()

    def tags_that_nouns_are_most_commonly_found_after(self):
        tag_list = []
        for i in range(len(self.word_tag)):
            try:
                a = self.word_tag[i + 1][1]
                if a.startswith('N'):
                    tag_list.append(self.word_tag[i][1])
            except:
                pass
        freq = FreqDist(tag_list)
        return freq.most_common()
    
    def proportion_ambiguous_word_types(self):
        word_tag_dict = {}
        for group in self.word_tag:
            word = group[0].lower()
            pos = group[1]
            if word in word_tag_dict:
                word_tag_dict[word].add(pos)
            else:
                word_tag_dict[word] = {pos}
        word_type = set([i.lower() for i in nltk.word_tokenize(self.raw)])
        total_len = len(word_type)
        word_len = 0
        for word in word_type:
            if len(word_tag_dict[word]) == 1:
                word_len += 1
        return 1 - word_len/total_len
    
    def proportion_ambiguous_word_tokens(self):
        word_tag_dict = {}
        for group in self.word_tag:
            word = group[0].lower()
            pos = group[1]
            if word in word_tag_dict:
                word_tag_dict[word].add(pos)
            else:
                word_tag_dict[word] = {pos}
        total_len = len(nltk.word_tokenize(self.raw))
        word_len = 0
        for word in nltk.word_tokenize(self.raw):
            a = word.lower()
            if len(word_tag_dict[a]) == 1:
                word_len += 1
        return 1 - word_len/total_len


if __name__ == "__main__":
    take_arg = sys.argv
    news = brown.tagged_sents(categories = 'news')

    if len(take_arg) == 2:
        method = take_arg[1]
        if 'train' in method:
            train_x, train_y = postag.to_dataset(news)
            model1 = postag.get_trained_model(train_x, train_y)
        else:
            print('Please check the input')       
    elif len(take_arg) == 3:
        train_x, train_y = postag.to_dataset(news)
        model1 = postag.get_trained_model(train_x, train_y)
        method = take_arg[1]
        argv = take_arg[2]
        if 'run' in method and isinstance(argv, str):
            sents = argv.split(' ')
            print(postag.get_pos_tag(model1, sents))
        elif 'test' in method:
            data_tag = brown.tagged_sents(categories = argv)
            test_x, test_y = postag.to_dataset(data_tag)
            print(model1.score(test_x, test_y))



