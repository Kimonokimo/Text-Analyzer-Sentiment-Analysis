import nltk
from nltk.corpus import brown as bn
from nltk.corpus import words
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.collections import Counter


COMPILED_BROWN = 'brown.pickle'



class BrownCorpus(object):

    def __init__(self):
        self.tagged_words = bn.tagged_words()
        self.words = bn.words()
    
    def __len__(self):
        return len(self.tagged_words)

    def __getitem__(self, i):
        return self.tagged_words[i]

    def __str__(self):
        name = 'Brown from nltk corpus' 
        return "<Text%s tokens=%s>" % (name, len(self))

def nouns_more_common_in_plural_form(corups):
    tag_words = corups.tagged_words
    words = corups.words
    words_count = Counter(words)
    plural_list = [i[0] for i in tag_words if i[0].endswith('s') and i[1].endswith('S')]
    plural_set = set(plural_list)
    out_list = []
    for words in plural_set:
        single = words[:-1]
        if words_count[words] > words_count[single]:
            out_list.append(words)
    return sorted(out_list)


def which_word_has_greatest_number_of_distinct_tags(corups):
    tag_words = corups.tagged_words
    word_tag_dict = {}
    for group in tag_words:
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


def tags_in_order_of_decreasing_frequency(corups):
    tag_words = corups.tagged_words
    tag_list = []
    for group in tag_words:
        pos = group[1]
        tag_list.append(pos)
    freq = FreqDist(tag_list)
    return freq.most_common()


def tags_that_nouns_are_most_commonly_found_after(corups):
    tag_words = corups.tagged_words
    tag_list = []
    for i in range(len(tag_words)):
        try:
            a = tag_words[i + 1][1]
            if a.startswith('N'):
                tag_list.append(tag_words[i][1])
        except:
            pass
    freq = FreqDist(tag_list)
    return freq.most_common()


def proportion_ambiguous_word_types(corups):
    tag_words = corups.tagged_words
    word_tag_dict = {}
    for group in tag_words:
        word = group[0].lower()
        pos = group[1]
        if word in word_tag_dict:
            word_tag_dict[word].add(pos)
        else:
            word_tag_dict[word] = {pos}
    word_type = set([i.lower() for i in bn.words()])
    total_len = len(word_type)
    word_len = 0
    for word in word_type:
        if len(word_tag_dict[word]) == 1:
            word_len += 1
    return (1 - (word_len/total_len))


def proportion_ambiguous_word_tokens(corups):
    tag_words = corups.tagged_words
    word_tag_dict = {}
    for group in tag_words:
        word = group[0].lower()
        pos = group[1]
        if word in word_tag_dict:
            word_tag_dict[word].add(pos)
        else:
            word_tag_dict[word] = {pos}
    total_len = len(tag_words)
    word_len = 0
    for word in bn.words():
        a = word.lower()
        if len(word_tag_dict[a]) == 1:
            word_len += 1
    return (1 - (word_len/total_len))