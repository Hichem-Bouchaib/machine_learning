import os
import numpy as np
import time
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import pandas
import csv
import collections, numpy

LENGTH_TRAINING=4000
LENGTH_TESTING=1000

def make_Dictionary(messages):
    all_words = []
    for line in messages:
            words = line.split()
            for word in words:
                if word!="ham" and word!="spam":
                    all_words += words
    dictionary = Counter(all_words)
    for item in list(dictionary): 
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(2500)
    return dictionary


def split_data(line_beginning,line_ending):
    messages = [line.rstrip() for line in open('messages.txt')]
    dataset = []
    for i, line in enumerate(messages):
        if i>=line_beginning and i<line_ending:
            words = line.split()
            dataset += words[1:]
    return dataset

def get_Y():
    #words=[]
    words = np.zeros(LENGTH_TRAINING+LENGTH_TESTING)
    i=0
    with open('messages.txt', 'r') as f:
        for line in f:
            if line.strip():
                if line.split(maxsplit=1)[0]=="ham":
                    words[i]=0
                if line.split(maxsplit=1)[0]=="spam":
                    words[i]=1
                i+=1          
    return words

def extract_features(beginning,length):
    messages = [line.rstrip() for line in open('messages.txt')]
    features_matrix = np.zeros((length-beginning, 2500))
    docID = 0
    for i, line in enumerate(messages):
        if i >=beginning and i<length:
            words = line.split()
            for word in words:
                wordID = 0
                for j, d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = j
                        features_matrix[docID, wordID] = words.count(word)
            docID = docID + 1
    return features_matrix

#Generation the training, testing set and Y_train/Y_test
training_set=split_data(0,LENGTH_TRAINING)
test_set=split_data(LENGTH_TRAINING,LENGTH_TRAINING+LENGTH_TESTING)
Y=get_Y()
Y_train=Y[0:LENGTH_TRAINING]
Y_test=Y[LENGTH_TRAINING:LENGTH_TRAINING+LENGTH_TESTING]

#Dictionnary of training set
dictionary=make_Dictionary(training_set)

#Generation of X_train and X_test
X_train=extract_features(0,LENGTH_TRAINING)
X_test=extract_features(LENGTH_TRAINING,LENGTH_TRAINING+LENGTH_TESTING)

# Training SVM and Naive bayes classifier and its variants. We tes test the time of the algorithm's execution
model2 = MultinomialNB()
start = time.time()
model2.fit(X_train, Y_train)
end = time.time()
print("Time for Naive Bayes : %s"%(end-start))

model1 = LinearSVC()
start2=time.time()
model1.fit(X_train, Y_train)
end2=time.time()
print("Time for SVM : %s"%(end2-start2))
print("")


#Then we check the score (accuracy)
print("Score for training set with SVM : %s"% model1.score(X_train,Y_train))
print("Score for training set with Naive Bayes : %s"%model2.score(X_train,Y_train))
print("Score for testing set with SVM : %s"% model1.score(X_test,Y_test))
print("Score for testing set with Naive Bayes : %s"%model2.score(X_test,Y_test))
print("");
# Test the unseen mails for Spam
result1 = model1.predict(X_test)
result2 = model2.predict(X_test)

#Print the confusion matrix for both of the algorithms
print("Confusion matrix with SVM")
print(confusion_matrix(Y_test, result1))
print("")
print("Confusion Matrix with Naive Bayes")
print(confusion_matrix(Y_test, result2))
