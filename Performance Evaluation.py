
""" To achieve the accuracy of the algorithm, we can compare the overall polarity with the average rate of
the customers."""

import pandas as pd
import numpy as np

df_polarity = pd.read_csv('Polarized Features.csv')
df_review_rated = pd.read_csv('Brooks England B17 - Second Version with Rates.csv')
features = {1 : 'leather', 2 : 'cover', 3 : 'gel', 4 : 'cushy', 5 : 'weight', 6 : 'feel', 7 : 'price', 8 : 'color'}



df_polarity_mean = pd.DataFrame(columns=features.values())
dff = []

for i in range(len(features.keys())):
    dff.append(np.mean(df_polarity.iloc[:, i]))
    # I don't why this doesn't work:    df_polarity_mean.loc[0, i] = np.mean(df_polarity.iloc[:, i])
df_polarity_mean.loc[0] = dff
#%%


df_rates = pd.DataFrame(columns=features.values())

for k in range(len(df_review_rated)):
    review = df_review_rated['content'][k]
    for feature in features.values():
        if feature in review:
            rate = df_review_rated['rating'][k]
            df_rates = df_rates.append({feature: rate}, ignore_index=True)



dfr = []
for l in range(len(features.values())):
    dfr.append(np.mean(df_rates.iloc[:, l]))
    # I don't why this doesn't work:
df_polarity_mean.loc[1] = dfr



#%%
input_start = -1
input_end = 1
output_start = 1
output_end = 5

def Polarity_to_Rate(input):
    slope = (output_end - output_start) / (input_end - input_start)
    output = output_start + slope * (input - input_start)
    return output

df_polarity_mean.loc[2] = df_polarity_mean.loc[0].apply(Polarity_to_Rate)

df_polarity_mean.loc[3] = 100 - \
                          (100 * ((df_polarity_mean.loc[1] - df_polarity_mean.loc[2]) / df_polarity_mean.loc[1]))


df_polarity_mean.to_csv('Performance Evaluation.csv')



# Let’s add a new row in above dataframe by passing dictionary i.e.
# # Pass the row elements as key value pairs to append() function
# modDfObj = dfObj.append({'Name' : 'Sahil' , 'Age' : 22} , ignore_index=True)
