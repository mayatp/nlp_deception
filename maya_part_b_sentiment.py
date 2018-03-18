#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  20 18:33:02 2018

@author: mayapetranova
"""

import csv    
import time    
import nltk.stem
import nltk
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support # to report on precision and recall
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

##################
## QUESTION 3 B ##
##################

print("Start of execution")
startTime = time.time() # Start time of the execution

# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Verified, Category, Label) = parseReview(line)
            rawData.append((Id, Text, Verified, Category, Label))
            preprocessedData.append((Id, preProcess(Text), Verified, Category, Label))
            if (Label == "Positive"):
                pos.append(1)
            else:
                neg.append(1)
        #print ('Positives', sum(pos))
        #print ('Negatives', sum(neg))
                
pos = []
neg = []

# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    # print (reviewLine[0]) # IDs
    # print (reviewLine[8]) # Review text
    # print (reviewLine[1]) # Labels - L1 -> Fake L2 -> Real
    # [0]DOC_ID	
    # [1]LABEL	
    # [2]RATING 
    # [3]VERIFIED_PURCHASE 
    # [4]PRODUCT_CATEGORY 
    # [5]PRODUCT_ID 
    # [6]PRODUCT_TITLE 
    # [7]REVIEW_TITLE 
    # [8]REVIEW_TEXT
    
    #print (reviewLine[0], reviewLine[8], reviewLine[1]) --> TESTING  
    
    Id = reviewLine[0]
    Text = reviewLine[8]
    Label = reviewLine[1]
    
    # Q3 adding more stuff - NEED TO CHOOSE 3
    Rating = reviewLine[2] 
    Verified = reviewLine[3]
    Category = reviewLine[4]
    if (Rating == '5'):
        Label = 'Positive'
    if (Rating == '4'):
        Label = 'Positive'
    if (Rating == '2'):
        Label = 'Negative'
    if (Rating == '1'):
        Label = 'Negative'
    if (Rating == '3'):
        Label = 'Negative'
    return (Id, Text, Verified, Category, Label)


# TEXT PREPROCESSING AND FEATURE VECTORIZATION
    
def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    
    for (_, Text, Verified, Category, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        Dictionary = toFeatureVector(preProcess(Text))
        Dictionary.update({'Verified':Verified, 'Category':Category}) # Adding Verified and Category as features
        trainData.append((Dictionary, Label))
    for (_, Text, Verified, Category, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        Dictionary = toFeatureVector(preProcess(Text))
        Dictionary.update({'Verified':Verified, 'Category':Category}) # Adding Verified and Category as features
        testData.append((Dictionary, Label))

def preProcess(text):

    tokenizer = RegexpTokenizer(r'[a-zA-Z]\w+\'?\w*')
    tokenizer.tokenize(text) #removing all type of punctuation

    english_stemmer = nltk.stem.SnowballStemmer('english')
    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda text: (english_stemmer.stem(w) for w in analyzer(text))
        
    stem_vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
    stem_analyze = stem_vectorizer.build_analyzer()
    myList = stem_analyze(text)
        
    return myList

globalFeatureDict = {} # If used, it affects the performance of the classifier

def toFeatureVector(tokens):
    
    localFeatureDict = {}
    
    for token in tokens:
        rmv_tokens = ['!).', ':', 'http', '.', ',', '?', '...', "'s", "n't", 'RT', ';', '&', ')', '(', '``', 'u', '(', "''", '|', '!']
        token = WordNetLemmatizer().lemmatize(token)
        
        #rmv_tokens = []

        if token not in rmv_tokens:
            if token[0:2] != '//':
                if isinstance(token, str):
                    if token not in localFeatureDict:
                        localFeatureDict[token] = 1
                        globalFeatureDict[token] = 1  # simulatnously building up a global dictinoary to keep track of number of features 
                    else:
                        localFeatureDict[token] += 1
                        globalFeatureDict[token] += 1
                          
    return localFeatureDict

# List definitions
rawData = []          # the filtered data from the dataset file (should be 21000 samples)
preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
data = []
trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'

# references to the data files
reviewPath = 'amazon_reviews.txt'

# MAIN

loadData(reviewPath)
splitData(0.8)

print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(globalFeatureDict), sep='\n')

# TRAINING AND VALIDATING OUR CLASSIFIER

def trainClassifier(trainData):
    
    pipeline = Pipeline([('svc', LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0))])
    return SklearnClassifier(pipeline).train(trainData)
    
# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))

def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))

def crossValidate(dataset, folds):
    shuffle(dataset)
    results = []
    foldSize = int(len(dataset)/folds) #it wants an int instead of float
    print ("Fold size:", foldSize)
    print("Training Classifier...")
    for i in range(0,len(dataset),foldSize):
        crossTestData = dataset[i:i+foldSize]
        crossTrainData = dataset[:i] + dataset[i+foldSize:]
        classifier = trainClassifier(crossTrainData)
        y_true = [x[1] for x in crossTestData]
        y_pred = predictLabels(crossTestData,classifier)
        results.append(precision_recall_fscore_support(y_true, y_pred, average='weighted'))

    return results
    
cv_results = crossValidate(trainData, 10) #10 fold as specified in the coursework spec
print("Classifier accuracy when using review rating as the sentiment gold standard")
print ("CV_SENTIMENT_RESULTS", cv_results)

print("End of execution")
endTime = time.time()
executionTime = endTime-startTime
print("Execution Time:", int(executionTime), " sec")


