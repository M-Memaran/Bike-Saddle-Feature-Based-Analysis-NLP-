

# Loading dataset

import pandas as pd

df00 = pd.read_csv('Brooks England B17 - Second Version.csv')


# Pre-processing

#1 Tokenizing


#2 Lemmatization using spaCy

import spacy
import time

# # example of using lemmatization by spacy
spcy_en = spacy.load('en_core_web_sm')      # If it doesn't work, try "python -m spacy download en" in the terminal
# string = "Hello! I don't know what I'm doing here. Whatttttt theeeee haappppyyyyness I've had. how'd'y"
# doc = spcy_en(string)
# lemmas = [token.lemma_ for token in doc]
# print(lemmas)


def lemmatization(review):
    new_review = spcy_en(str(review))
    lemmas_review = [token.lemma_ for token in new_review]
    return lemmas_review

start_time = time.time()
df01 = df00.content.apply(lambda review: lemmatization(review))
end_time = time.time()
print(df00.head())
print(df01.head())
print(df01.describe())
print('Time of lemmatization execution: ', end_time - start_time)

#%%
# convert to lower case

def convert_to_lowercase(review):

    for i in range(len(review)):
        review[i] = review[i].lower()
    return review

start_time = time.time()
df02 = df01.apply(lambda review: convert_to_lowercase(review))
end_time = time.time()
print(df01.describe())
print('Time of converting to lover case: ', end_time - start_time)

#%%
# removing punctuation

import re
import string

def eliminate_punctuation(review, regex):
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    return new_review


regex = re.compile('[%s]' % re.escape(string.punctuation))

start_time = time.time()
df03 = df02.apply(lambda review: eliminate_punctuation(review, regex))
end_time = time.time()
print(df01.describe())
print('Time of elimination of punctuation: ', end_time - start_time)

#%%
# correction

import re

from nltk.corpus import wordnet
# The class will replace the words like happyyyyyy into happy by
# by checking the word in synset.
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r"(.)\1{2,}")
        self.repl = r'\1'
    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


reprep = RepeatReplacer()

string = "Hello! I don't know what I'm doing here. Whatttttt theeeee haappppyyyyness I've had. how'd'y"
print(reprep.replace(string))

start_time = time.time()
df04 = df03.apply(lambda review: reprep.replace(str(review)))
end_time = time.time()
print(df04.describe())
print('Time of correction: ', end_time - start_time)

#%%

# remove numbers

def remove_number(text):
    return ''.join(c for c in text if not c.isdigit())

start_time = time.time()
df05 = df04.apply(lambda review: remove_number(review))
end_time = time.time()
print(df05.describe())
print('Time of remove number: ', end_time - start_time)

print(df00.head())
print(df05.head())

#%%

def convert_text_to_list(review):
    return review.replace("[","").replace("]","").replace("'","").replace("\t","").split(",")

# Convert "reviewText" field to back to list
df06 = df05.astype(str)
df06 = df06.apply(lambda text: convert_text_to_list(text));
print(df06.head())

# Stopwords elimination
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(review):
    return [token for token in review if token not in stop_words]

start_time = time.time()
df07 = df06.apply(lambda review: remove_stopwords(review))
end_time = time.time()
print(df07.describe())
print('Time of remove stopwords: ', end_time - start_time)


#%%
# Frequent words extraction

from nltk.probability import FreqDist

def collect_zipfs_law_metrics(review, fd):
    for token in review:
        fd.update([token])



fd = FreqDist()
df07.apply(lambda review: collect_zipfs_law_metrics(review, fd));

# print(fd)


words = []
freqs = []



for rank, word in enumerate(fd):
    words.append(word)
    freqs.append(fd[word])



frequencies = {'word': words, 'frequency':freqs}
frequencies_df = pd.DataFrame(frequencies)



print(frequencies_df.head())



frequencies_df = frequencies_df.sort_values(['frequency'], ascending=[False])
frequencies_df = frequencies_df.reset_index()
frequencies_df = frequencies_df.drop(columns=['index'])


print(frequencies_df[0:20])

#%%
# Part of speech (PoS)

"""
The list of all possible tags appears below:

| Tag  | Description                              |
|------|------------------------------------------|
| CC   | Coordinating conjunction                 |
| CD   | Cardinal number                          |
| DT   | Determiner                               |
| EX   | ExistentialĘthere                        |
| FW   | Foreign word                             |
| IN   | Preposition or subordinating conjunction |
| JJ   | Adjective                                |
| JJR  | Adjective, comparative                   |
| JJS  | Adjective, superlative                   |
| LS   | List item marker                         |
| MD   | Modal                                    |
| NN   | Noun, singular or mass                   |
| NNS  | Noun, plural                             |
| NNP  | Proper noun, singular                    |
| NNPS | Proper noun, plural                      |
| PDT  | Predeterminer                            |
| POS  | Possessive ending                        |
| PRP  | Personal pronoun                         |
| PRP* | Possessive pronoun                       |
| RB   | Adverb                                   |
| RBR  | Adverb, comparative                      |
| RBS  | Adverb, superlative                      |
| RP   | Particle                                 |
| SYM  | Symbol                                   |
| TO   | to                                       |
| UH   | Interjection                             |
| VB   | Verb, base form                          |
| VBD  | Verb, past tense                         |
| VBG  | Verb, gerund or present participle       |
| VBN  | Verb, past participle                    |
| VBP  | Verb, non-3rd person singular present    |
| VBZ  | Verb, 3rd person singular present        |
| WDT  | Wh-determiner                            |
| WP   | Wh-pronoun                               |
| WP*  | Possessive wh-pronoun                    |
| WRB  | Wh-adverb                                |

Notice: where you see `*` replace with `$`.
"""
import nltk
frequencies_df['Tag of word'] = nltk.pos_tag(frequencies_df['word'])
print(frequencies_df.head())

frequencies_df.to_csv('Brooks England B17 - Frequent Words.csv')
# frequencies_df.to_csv('Brooks England B17 - Frequent Words.csv', sep='\t', header=True, index=False);

# Save a dictionary into a pickle file.
frequencies_df.to_pickle('Brooks England B17 - Frequent Words.p')




