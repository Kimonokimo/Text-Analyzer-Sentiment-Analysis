# Text Analyzer

In this class we will build a text analyzer, starting from scratch and adding more and more functionality as the course goes on. This repository contains the instructions for the first step. For later assignments we will be given additional instructions, data and code that can be added to this repository.

[ [assignment 3](extensions/a3-search.md)
| [assignment 4](extensions/a4-tags.md)
| [assignment 5](extensions/a5-sentiment.md)
]

We start of with this readme file, a `data` directory and a `main.py` script. My mission is to edit the main script and add the following functionality:

1. Read source data from a file or a set of files in a directory
2. Generate some simple statistics for the source data
3. Generate a vocabulary for the source data
4. Compare the source data to categories in the Brown corpus

In general, unless specifically stated otherwise, we are allowed to:

- use any standard Python module
- use any class or function from the nltk package

We are not allowed to use any other third party module.

This repository also includes a file `test.py` with unit tests, which will be the same test file that we will use. 


## Reading source data

There are three data sources in the `data` directory:

```
grail.txt  -  Monty Python and the Holy Grail
emma.txt   -  Emma by Jane Austen
wsj/*      -  the first 25 documents of the Wall Street Journal corpus
```

We need to finish the `read_text()` method so that it returns an instance of `nltk.text.Text`. 

```
>>> read_text('data/emma.txt')
<Text: Emma by Jane Austen 1816>
```

Reference:
- [Chapter 2](https://www.nltk.org/book/ch02.html) of the NLTK book shows how to load a corpus.



## Generate simple statistics

For the text find the total number of word tokens, word types and sentences. Do this by finishing `token_count()`, `type_count()` and `sentence_count()`. For the type count, we should at least normalize case so that 'the' and 'The' are counted as the same type. 

When we have finished the three functions we are able to do something like the following:

```
>>> emma = read_text('data/emma.txt')
>>> token_count(emma)
191673
>>> type_count(emma)
8000
>>> sentence_count(emma)
8039
```

Our counts may differ from the ones above since your code may be slightly different from mine, but they should be in the same ball park though. This holds for all following examples.

In addition, edit and complete the following two functions:

`most_frequent_content_words()`. Return a list with the 25 most frequent content words and their frequencies. The list should have (word, frequency) pairs and be ordered on the frequency. We used the stop word list in `nltk.corpus.stopwords` in our definition of what a content word is.

```
>>> most_frequent_content_words(emma)[:5]
[('Mr.', 1089), ('Emma', 855), ('could', 824), ('would', 813), ('Mrs.', 668)]
```

`most_frequent_bigrams()`. Return a list with the 25 most frequent bigrams that only contain content words. The list returned have pairs where the first element in the pair is the bigram and the second the frequency, as in ((word1, word2), frequency), these are ordered on frequency.

```
>>> most_frequent_bigrams(emma)[:5]
[(('Mr.', 'Knightley'), 271),
 (('Mrs.', 'Weston'), 246),
 (('Mr.', 'Elton'), 211),
 (('Miss', 'Woodhouse'), 168),
 (('Mr.', 'Weston'), 158)]
````


## Generate a vocabulary for the text

A vocabulary is an object that we here create from scratch and that allows us to do a couple of things. We create it from an instance of Text and it should contain a list (or set) of all words in a text and some other information. The important thing is that we set up our class in such a way that we can easily find some minimal information on a word and print a concordance.

The minimal information for the word are the frequency in the text, the most likely part of speech and a description of the word's most likely meaning (that is, the WordNet gloss).

```
>>> vocab = Vocabulary(read_text('data/grail.txt'))
>>> vocab.frequency('swallow'))
10
>>> vocab.pos('swallow'))
'n'
>>> vocab.gloss('swallow'))
'a small amount of liquid food'
```

In addition, it should allow you to print a concordance for a word.

```
>>> vocab.kwic('swallow')
Displaying 10 of 10 matches:
 is a temperate zone . ARTHUR : The swallow may fly south with the sun or the h
be carried . SOLDIER # 1 : What ? A swallow carrying a coconut ? ARTHUR : It co
 to maintain air-speed velocity , a swallow needs to beat its wings forty-three
: It could be carried by an African swallow ! SOLDIER # 1 : Oh , yeah , an Afri
OLDIER # 1 : Oh , yeah , an African swallow maybe , but not a European swallow
 swallow maybe , but not a European swallow . That 's my point . SOLDIER # 2 :
 and Sir Bedevere , not more than a swallow 's flight away , had discovered som
something . Oh , that 's an unladen swallow 's flight , obviously . I mean , th
he air-speed velocity of an unladen swallow ? ARTHUR : What do you mean ? An Af
o you mean ? An African or European swallow ? BRIDGEKEEPER : Huh ? I -- I do n'
```

Our vocabulary is restricted to words in the NLTK Words corpus (see section 4.1 in https://www.nltk.org/book/ch02.html).


##  Comparing the text to the Brown corpus

The last part of the assignment is to turn a Text into a vector and compare it to the vectors for five categories in the Brown corpus: *adventure*, *fiction*, *government*, *humor* and *news* (the words for these categories we can get using NLTK). Compare the text vector to the category vector using the cosine measure to get the similarity of the two vectors:

- For a text, take all the words in the text, create a frequency dictionary and use it to create a vector.
- For a category, take all the words in the category and do the same as above.

The problem with this is that these vectors have different dimensions and you cannot directly compare them. We will have to decide on how many dimensions to use. 

It is probably a good idea to reuse and/or extend the `Vocabulary` class.

```
>>> grail = read_text('data/grail.txt')
>>> compare_to_brown(grail)
adventure    0.84
fiction      0.83
government   0.79
humor        0.87
news         0.82
```

The similarity measure of the grail text and the Brown categories should be printed to the standard output. Run the same comparison with Emma and the Wall Street Journal data. 

This may be the part of the assignment that has the slowest running time (although the orinial imports and pinging WordNet for the first time also take several seconds). However, it should not require more running time than a few dozen second or maybe up to a minute on a slower machine.
