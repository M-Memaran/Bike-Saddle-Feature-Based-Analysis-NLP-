
import pandas as pd
import re
import string
import operator
from collections import OrderedDict
from textblob import TextBlob
from nltk.corpus import stopwords
import os
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np


# Dict to convert the raw user text to meaningful words for analysis
apostropheList = {"n't" : "not","aren't" : "are not","can't" : "cannot","couldn't" : "could not","didn't" : "did not",\
                  "doesn't" : "does not", "don't" : "do not","hadn't" : "had not","hasn't" : "has not","haven't" : \
                  "have not","he'd" : "he had","he'll" : "he will", "he's" : "he is","I'd" : "I had","I'll" : "I will",\
                  "I'm" : "I am","I've" : "I have","isn't" : "is not","it's" : "it is","let's" : "let us","mustn't" : \
                  "must not","shan't" : "shall not","she'd" : "she had","she'll" : "she will", "she's" : "she is", \
                  "shouldn't" : "should not","that's" : "that is","there's" : "there is","they'd" : "they had", \
                  "they'll" : "they will", "they're" : "they are","they've" : "they have","we'd" : "we had","we're" : \
                  "we are","we've" : "we have", "weren't" : "were not", "what'll" : "what will","what're" : "what are",\
                  "what's" : "what is","what've" : "what have", "where's" : "where is","who'd" : "who had", "who'll" : \
                  "who will","who're" : "who are","who's" : "who is","who've" : "who have", "won't" : "will not", \
                  "wouldn't" : "would not", "you'd" : "you had","you'll" : "you will","you're" : "you are","you've" : \
                  "you have"}


# Removing stop words might lead to better data analysis
stopWords = stopwords.words("english")

# Exclude punctuations from the reviews
exclude = set(string.punctuation)


df = pd.read_csv('Brooks England B17 - Second Version.csv')
features = {1 : ('leather', 'cover', 'gel', 'cushy'), 2 : ('weight'), 3 : ('feel'), 4 : ('price'), 5 : ('color')}
df_polarity = pd.DataFrame(columns = features.keys())
adjectives = []


# function to return key for any value
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"



def get_overal_polarity(excerpts):
    text = ''.join(excerpts)
    blob = TextBlob(text)

    polarity = []
    for sentence in blob.sentences:
        polarity.append(sentence.sentiment.polarity)

    return np.mean(polarity)


review = "I have weight like a woman. But, It isn't so heavy."
def feature_polarity(review, features):

    review_sentences = nltk.sent_tokenize(review)
    for sen_index in range(len(review_sentences)):
        sentence = review_sentences[sen_index]
        words = nltk.word_tokenize(sentence)
        for w_index in range(len(words)):
            word = words[w_index]
            if word in features.values():
                feature_key = get_key(word, features)
                polarity = get_overal_polarity(sentence)
                return df_polarity.append({ feature_key : polarity}, ignore_index=True)



##############################################  New Version  ##########################################################
########################################################################################################################
#%%

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize


def get_overal_polarity(excerpts):
    text = ''.join(excerpts)
    blob = TextBlob(text)

    polarity = []
    for sentence in blob.sentences:
        polarity.append(sentence.sentiment.polarity)

    return np.mean(polarity)

reviewContent = []
df00 = pd.read_csv('Brooks England B17 - Second Version.csv')


print(get_overal_polarity("I don't like this"))
print(get_overal_polarity("I really hate this"))



# for i in range(len(df)):
#     review = df.loc[i]
#     reviewContent.append(review)


# pair = nltk.sent_tokenize(reviewContent[3])

# print(len(pair.split()))
# print(pair)



#%%

features = {1 : ('leather', 'cover', 'gel', 'cushy'), 2 : ('weight'), 3 : ('feel'), 4 : ('price'), 5 : ('color')}
print(features[5])

df = pd.DataFrame(columns = features.keys())
df = df.append({ 1 : 23}, ignore_index=True)
df = df.append({ 1 : 45}, ignore_index=True)
df = df.append({ 1 : 10}, ignore_index=True)
df = df.append({ 1 : -12}, ignore_index=True)
df = df.append({ 1 : -23}, ignore_index=True)

print(np.mean(df.loc[:, 2]))


# for i in features.keys():
    # features_polarity(i) = []


# df1 = pd.DataFrame(data = , columns = features.keys())

# def update_polarity(feature_key, polarity):
#     feature_polarity(feature_key)
#%%

review = 'As a bigger guy ( 230lbs ) I ride about 200 miles per week and am always looking to ride longer. I have" \
                                " tried a fair amount of saddles including the super fancy selle italia flite gel flow" \
                                " max which was OK for about a year. After about 3000 miles, it started to hurt more " \
                                "and more despite my weight dropping and extensive seat time. I have always heard these " \
                                "saddles were the most comfortable so I took the risk and IT WAS THE GREATEST CHOICE OF" \
                                " MY LIFE! From the first moment, there was a huge comfort increase. It felt like I was " \
                                "sitting on the sofa. On the first day, I rode it for 30 miles with no padded shorts to" \
                                " help break it in (Which was super comfortable) then the next day I rode 102 miles with" \
                                " padded shorts. The century was the most comfortable I have ever been on a bike and it " \
                                "is getting better. I wish I had bought this in the very beginning instead of looking " \
                                "for the sleek saddles that hurt. I have already gotten multiple compliments on its" \
                                " cools looks and find myself answering other peoples questions on how comfortable it" \
                                " is.'

review_sentences = nltk.sent_tokenize(review)
print(type(review))
print(type(review_sentences))

for i in range(len(review_sentences)):
    print(review_sentences[i])

#%%
features = {1 : 'leather', 2 : 'cover', 3 : 'gel', 4 : 'cushy', 5 : 'weight', 6 : 'feel', 7 : 'price', 8 : 'color'}
df_polarity = pd.DataFrame(columns = features.keys())
adjectives = []


# function to return key for any value
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"

# review = "I have weight like a woman. But, It isn't so heavy."
# def feature_polarity(review, features):
""" df00.loc[[0]]  or df00.iloc[0,0] ====> df00.iloc[0,0] get correct result"""


for j in range(len(df00)):
    review_sentences = nltk.sent_tokenize(df00.iloc[j,0])
    last_sentence_index = len(review_sentences)
    for sen_index in range(len(review_sentences) -1):
        sentence = review_sentences[sen_index] + review_sentences[sen_index + 1]
        # print(sentence)
        words = nltk.word_tokenize(sentence)
        # print(words)
        for w_index in range(len(words)):
            word = words[w_index]
            if word in features.values():
                # print(word)
                feature_key = get_key(word, features)
                polarity = get_overal_polarity(sentence)
                # print(polarity)
                df_polarity = df_polarity.append({ feature_key : polarity}, ignore_index=True)


#%%










