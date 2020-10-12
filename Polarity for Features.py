
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk



df = pd.read_csv('Brooks England B17 - Second Version.csv')
features = {1 : 'leather', 2 : 'cover', 3 : 'gel', 4 : 'cushy', 5 : 'weight', 6 : 'feel', 7 : 'price', 8 : 'color'}
df_polarity = pd.DataFrame(columns = features.keys())



# function to return key for any value
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"



# function to return overal polarity of an excerpts
""" Through some test, it is realized that this function can distinguish between different state words like "like" or "love"
"don't like" or "hate". Then it is not needed to score these words in a special group (like what some algorithms did that)"""

def get_overal_polarity(excerpts):
    text = ''.join(excerpts)
    blob = TextBlob(text)

    polarity = []
    for sentence in blob.sentences:
        polarity.append(sentence.sentiment.polarity)

    return np.mean(polarity)



# Finding the features in the sentences of the reviews and getting the polarity of the sentences in which
# there is the feature

for j in range(len(df)):
    review_sentences = nltk.sent_tokenize(df.iloc[j,0])
    last_sentence_index = len(review_sentences)
    for sen_index in range(len(review_sentences) -1):
        sentence = review_sentences[sen_index] + review_sentences[sen_index + 1]
        # By checking some reviews the opinion about some features is commented in the next sentence.
        words = nltk.word_tokenize(sentence)
        for w_index in range(len(words)):
            word = words[w_index]
            if word in features.values():
                feature_key = get_key(word, features)
                polarity = get_overal_polarity(sentence)
                df_polarity = df_polarity.append({ feature_key : polarity}, ignore_index=True)



df_polarity.to_csv('Polarized Features.csv', index = False)

#%%




