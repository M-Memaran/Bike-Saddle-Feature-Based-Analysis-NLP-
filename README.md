# Bike-Saddle-Feature-Based-Analysis-NLP-

This manual shows how to use the NLP process, which is created to extract the common features of a product on 
the Amazon website. It uses reviews of customers about the product. The mention features are the ones that are most 
commented on. By using polarity evaluation, the sentiment of customers is driven, and it is considered as a way of 
investigating the product feature for future developments. 
It should be noted; this work is categorized on feature-based analysis, sentiment analysis, feature-opinion extraction, 
or other lexical analyses. NLP, data mining, and different machine learning algorithms can be used to do these analyses. 
This work is a simple way, and it is a part of a thesis, and it is not the focus of the thesis. So, it can be improved, 
like many other processes. 


Follow this step: 

1- Do the code in the "Bike Saddle Reviews Scraping" folder to achieve the "data" file (customer reviews).  

2- Since the all data in "data" file is not needed, clean the 'data' file manually in excel to reach 
the file like "Brooks England B17 - Second Version with Rates" or "Brooks England B17 - Second Version". 
The first is better, and it would be used when the rate of the reviews is coming to use. 

3- Implement the "Feature Extraction" file. "Brooks England B17 - Frequent Words" file is the outcome of 
the implementation. But it should be checked and monitored manually (mapping) to be able to drive features. 

4- "Brooks England B17 - Features from Frequent Nouns" is the file that separated the features by humans. 
The features are inserted in the next file by the programmer as the features dictionary. 

5- Run the "Polarity for Features" file and find the results in the "Polarized Features" file. 

6- Run "Performance Evaluation" and see the "Performance Evaluation" file to compare the NLP process 
results with the results of rates and check the performance of the NLP model.

*** "Performance Evaluation - Excel" shows the performance analysis visually.    
