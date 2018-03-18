#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:03:11 2018

@author: mayapetranova
"""
import pandas as pd
import csv
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from statistics import mean
from nltk.corpus import stopwords
from textstat.textstat import textstat
from collections import Counter

##################
## QUESTION 1 B ##
##################

file_path = 'amazon_reviews.txt'
amazon_data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, header=None,
            names = ['DOC_ID', 'LABEL', 'RATING', 'VERIFIED_PURCHASE',
            'PRODUCT_CATEGORY', 'PRODUCT_ID', 'PRODUCT_TITLE', 'REVIEW_TITLE', 'REVIEW_TEXT'],
            na_values='?')

#amazon_data.info() # checking the data types before changing them to integers

# Label encoding
convert_to_nums = {"LABEL": {"__label2__": 1, "__label1__": 0},
                "VERIFIED_PURCHASE": {"Y": 1, "N": 0},
                "PRODUCT_CATEGORY": {"Apparel" : 1, "Automotive" : 2, "Baby" : 3, "Beauty" : 4,
                                     "Books" : 5, "Camera" : 6, "Electronics" : 7, "Furniture" : 8,
                                     "Grocery" : 9, "Health" : 10, "Home" : 11, "Jewelry" : 12,
                                     "Kitchen" : 13, "Lawn" : 14, "Luggage" : 15, "Musical" : 16, 
                                     "Office" : 17, "Outdoors" : 18, "PC" : 19, "Pet" : 20, "Shoes" : 21,
                                     "Sports" : 22, "Tools" : 23, "Toys" : 24, "Video" : 25, 
                                     "Watches" : 26,"Wireless" : 27 }}

amazon_data.replace(convert_to_nums, inplace=True)
#print(amazon_data.dtypes) # changed datatypes for the features we need

normalised_data = amazon_data[['LABEL', 'RATING', 'VERIFIED_PURCHASE', 'PRODUCT_CATEGORY' ]].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler() # using scikit learn to normalise the numerical data into value between 0 and 1
x_scaled = min_max_scaler.fit_transform(normalised_data)
amazon_normalised = pd.DataFrame(x_scaled)

amazon_normalised.columns = ['F_R', 'RATE', 'VERIF', 'CATEG'] #giving the columns names for easier comparison later as after the normalisation they turned to 0,1,2,3

print("Corr b-n fake/real and rating: ", amazon_normalised['F_R'].corr(amazon_normalised['RATE']))
print("Corr b-n fake/real and verified: ", amazon_normalised['F_R'].corr(amazon_normalised['VERIF']))
print("Corr b-n fake/real and category: ", amazon_normalised['F_R'].corr(amazon_normalised['CATEG']))

fr_rate = amazon_normalised['F_R'].corr(amazon_normalised['RATE'])
fr_verif = amazon_normalised['F_R'].corr(amazon_normalised['VERIF'])
fr_categ = amazon_normalised['F_R'].corr(amazon_normalised['CATEG'])


##################
## QUESTION 2 B ##
##################

stop_words = stopwords.words('english') # Defining a list of English stop words

def parseReview(reviewLine):
    
    Id = reviewLine[0]
    Text = reviewLine[8]
    Label = reviewLine[1]
    
    Rating = reviewLine[2]
    Verified = reviewLine[3]
    Category = reviewLine[4]
    
    return (Id, Text, Rating, Verified, Category, Label)

def loadData(file_path, Text=None):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Rating, Verified, Category, Label) = parseReview(line)
            rawData.append((Id, Text, Rating, Verified, Category, Label))
            preprocessedData.append((Id, Text, Rating, Verified, Category, Label))
            if (Label == "__label2__"): # if the review is REAL, append to the realRevLen list 
                real.append(1)
                realRevLen.append(len(Text)) # add graph
                realReadEase.append(textstat.flesch_reading_ease(Text))
                realSylCnt.append(textstat.syllable_count(Text))
                realSentCnt.append(textstat.sentence_count(Text))
                realGrade.append(textstat.flesch_kincaid_grade(Text))
                realFog.append(textstat.gunning_fog(Text))
                realSmog.append(textstat.smog_index(Text))
                realAutoRead.append(textstat.automated_readability_index(Text))
                realDiffic.append(textstat.difficult_words(Text))
                realCat.append(Category)
                for word in Text:
                    if word in stop_words:
                        realStopNum.append(1) # number of stop words
                    if word in string.punctuation:
                        realPunctNum.append(1) # number of punctuation
                    if word.isupper(): # checking any capitals within the text (even within words)
                        realUpperNum.append(1) # number of uppercases
            else: # when the review is fake
                fake.append(1)
                fakeRevLen.append(len(Text))
                fakeReadEase.append(textstat.flesch_reading_ease(Text))
                fakeSylCnt.append(textstat.syllable_count(Text))
                fakeSentCnt.append(textstat.sentence_count(Text))
                fakeGrade.append(textstat.flesch_kincaid_grade(Text))
                fakeFog.append(textstat.gunning_fog(Text))
                fakeSmog.append(textstat.smog_index(Text))
                fakeAutoRead.append(textstat.automated_readability_index(Text))
                fakeDiffic.append(textstat.difficult_words(Text))
                fakeCat.append(Category)
                for word in Text:
                    if word in stop_words:
                        fakeStopNum.append(1)
                    if word in string.punctuation:
                        fakePunctNum.append(1)
                    if word.isupper(): 
                        fakeUpperNum.append(1)
                
        realNumRev = sum(real)
        fakeNumRev = sum(fake)
        print("Avg len of Real: ", round(mean(realRevLen)))
        print("Avg len of Fake: ", round(mean(fakeRevLen))) # rounding up to the 2nd floating point. If 2 is omitted it will round to a whole number
        print("Avg Real Stop words", (sum(realStopNum))/realNumRev) 
        print("Avg Fake Stop words", (sum(fakeStopNum))/fakeNumRev)
        print("Avg Real Punctuation", (sum(realPunctNum))/realNumRev) 
        print("Avg Fake Punctuation", (sum(fakePunctNum))/fakeNumRev)
        print("Avg Real Uppercases", sum(realUpperNum)) 
        print("Avg Fake Uppercases", sum(fakeUpperNum))
        
        # Flesch-Kincaid average scores for different measures
        ### READING EASE ###
        print("*********************************")
        print("Flesch-Kincaid Readability tests: ")
        print("*********************************")
        print("Avg Real Reading ease", round(mean(realReadEase))) # round(answer, 2) to the 2nd decimal
        print("Avg Fake Reading ease", round(mean(fakeReadEase)))
        print("Avg Real Syllable count", round(mean(realSylCnt))) 
        print("Avg Fake Syllable count", round(mean(fakeSylCnt)))
        print("Avg Real Sentence count", round(mean(realSentCnt))) 
        print("Avg Fake Sentence count", round(mean(fakeSentCnt)))
        print("Avg Real Grade", round(mean(realGrade),3)) 
        print("Avg Fake Grade", round(mean(fakeGrade),3))
        print("Avg Real Fog", round(mean(realFog),3)) 
        print("Avg Fake Fog", round(mean(fakeFog),3))
        print("Avg Real Smog", round(mean(realSmog),3)) 
        print("Avg Fake Smog", round(mean(fakeSmog),3))
        print("Avg Real Auto read", round(mean(realAutoRead),3)) 
        print("Avg Fake Auto read", round(mean(fakeAutoRead),3))        
        print("Avg Real Difficulty", round(mean(realDiffic),3)) 
        print("Avg Fake Difficulty", round(mean(fakeDiffic),3))

        
        #print("Total Real", sum(real)) # just checking - total reviews labelled "Real" = 10500
        #print("Total Fakes", sum(fake)) # total "Fake" = 10500
        
        #80-89 : Easy --> Fake ones are Easy
        #70-79 : Fairly Easy --> Read ones are Fairly easy    
        
rawData = []
preprocessedData = []

# real reviews lists definition
real = []
realRevLen = []
realReadEase = []
realSylCnt = []
realSentCnt = []
realGrade = []
realFog = []
realSmog = []
realAutoRead = []
realDiffic = []
realStopNum = []
realPunctNum = []
realUpperNum = []
realCat = []

# fake reviews lists definition
fake = []
fakeRevLen = []
fakeReadEase = []
fakeSylCnt = []
fakeSentCnt = []
fakeGrade = []
fakeFog = []
fakeSmog = []
fakeAutoRead = []
fakeDiffic = []
fakeStopNum = []
fakePunctNum = []
fakeUpperNum = []
fakeCat = []

loadData(file_path)

#fakeGroups = Counter(fakeRevLen)
#print(fakeGroups) # Shortest and longest fake review - 150; Longest - 2036 - help for defining the boundaries of the plots
#realGroups = Counter(realRevLen) 
#print(realGroups) # Shortest and longest real review - 197; Lonest- 4081

######################
## P L O T T I N G ###
######################

#Q1B
# plotting correlation strength fake/real VS Rating, Verified, Category
bins = ('Rating', 'Verified', 'Category')
y = np.arange(len(bins))
features = [fr_rate,fr_verif,fr_categ]
plt.bar(y, features, align='center', alpha=0.5)
plt.xticks(y, bins)
plt.ylabel('Correlation')
plt.title('Correlation strength')
plt.show()

#Q2B
# plotting review length distibution
realDist = plt.hist(realRevLen, bins=np.arange(4100), histtype='step', linewidth=1);
plt.ylabel('Frequency')
plt.xlabel('Review length (number of words)')
plt.title('Real review length distribution')
plt.show()

fakeDist = plt.hist(fakeRevLen, bins=np.arange(2050), histtype='step', linewidth=1);
plt.ylabel('Frequency')
plt.xlabel('Review length (number of words)')
plt.title('Fake review length distribution')
plt.show()

realDist = plt.hist(realRevLen, bins=np.arange(4100), histtype='step', linewidth=1);
fakeDist = plt.hist(fakeRevLen, bins=np.arange(2050), histtype='step', linewidth=1);
plt.ylabel('Frequency')
plt.xlabel('Review length (number of words)')
plt.title('Real(blue) vs Fake(orange) review distribution')
plt.show()

