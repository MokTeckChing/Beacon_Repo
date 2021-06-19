# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:45:26 2021

@author: zhou_bing_dian
"""
import nltk
import csv
import re

# We need this dataset in order to use the tokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Also download the list of stopwords to filter out
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = word_tokenize(text)

    # Remove the stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in stopwords.words('english')
    ]
    
    #print(tokenized_text)

    # Remember, this final output is a list of words
    return clean_text

#comments=[row for row in csv.reader(open(r'C:\Users\Zhou__5\Desktop\Y4S2\ST4299\New folder\nlp\Combined Well.csv'))]
comments=[row for row in csv.reader(open(r'C:\Users\Zhou__5\Desktop\Y4S2\ST4299\New folder\nlp\original ict dataset (WELL).csv'))]
#print (comments)

# Remove the first row, since it only has the labels
comments = comments[1:]
#comments

texts = [row[0] for row in comments]
#texts 

topics = [row[1] for row in comments]
#topics

# Process the texts to so they are ready for training
# But transform the list of words back to string format to feed it to sklearn
texts = [" ".join(process_text(text)) for text in texts]
print(texts)

#Countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
vectors = matrix.fit_transform(texts).toarray()


#TFIDF
'''
from sklearn.feature_extraction.text import TfidfVectorizer
matrix = TfidfVectorizer(max_features=1000)
vectors = matrix.fit_transform(texts).toarray()
# feature_names = matrix.get_feature_names()
# dense = vectors.todense()
# denselist=dense.tolist()
'''
#TFIDF ngram
'''
from sklearn.feature_extraction.text import TfidfVectorizer
matrix = TfidfVectorizer(ngram_range=(2,3), max_features=1000)
matrix.fit(df[TEXT])
vectors = matrix.fit_transform(texts).toarray()
'''

#split dataset

from sklearn.model_selection import train_test_split
vectors_train, vectors_test, topics_train, topics_test = train_test_split(vectors, topics, random_state=123)

#NAIVE BAYES
print("Classifier: GaussianNB")
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(vectors_train, topics_train)

topics_pred = classifier.predict(vectors_test)
from sklearn.metrics import classification_report
print(classification_report(topics_test, topics_pred))


#MULTINOMIAL NAIVE BAYES
print("Classifier: MultinomialNB")
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vectors_train, topics_train)

topics_pred = classifier.predict(vectors_test)
from sklearn.metrics import classification_report
print(classification_report(topics_test, topics_pred))

#KNN
print("Classifier: KNN")
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier()
classifier.fit(vectors_train, topics_train)
topics_pred = classifier.predict(vectors_test)
from sklearn.metrics import classification_report
print(classification_report(topics_test, topics_pred))

#RANDOM FOREST
print("Classifier: Random Forest")
from sklearn.ensemble import RandomForestClassifier as rf
classifier = rf()
classifier.fit(vectors_train, topics_train)

topics_pred = classifier.predict(vectors_test)
from sklearn.metrics import classification_report
print(classification_report(topics_test, topics_pred))


#LogisticRegression
print("Classifier: Logistic Regression")
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(vectors_train, topics_train)
topics_pred = classifier.predict(vectors_test)
from sklearn.metrics import classification_report
print(classification_report(topics_test, topics_pred))


#SVC
print("Classifier: SVC")
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(vectors_train, topics_train)
topics_pred = classifier.predict(vectors_test)
from sklearn.metrics import classification_report
print(classification_report(topics_test, topics_pred))




# #SVR
# print("Classifier: SVR")
# from sklearn.svm import SVR
# classifier = SVR()
# classifier.fit(vectors_train, topics_train)
# topics_pred = classifier.predict(vectors_test)
# from sklearn.metrics import classification_report
# print(classification_report(topics_test, topics_pred))

# #SVC linear
# print("Classifier: LinearSVC")
# from sklearn.svm import LinearSVC
# classifier = LinearSVC()
# classifier.fit(vectors_train, topics_train)
# topics_pred = classifier.predict(vectors_test)
# from sklearn.metrics import classification_report
# print(classification_report(topics_test, topics_pred))




# # Predict with the testing set
# topics_pred = classifier.predict(vectors_test)
# #print(topics_pred)
# #import numpy
# #topics_pred.to_csv(r'C:\Users\zhou_bing_dian\desktop\QUAL CODING\Areas Done Well Q8.csv', sep='\t')
# #numpy.savetxt(r'C:\Users\Zhou__5\Desktop\Y4S2\ST4299\New folder\QUAL CODING\Areas Done Well Q8.csv', topics_pred, delimiter=",", fmt = '%s')

# # measure the accuracy of the results
# from sklearn.metrics import classification_report
# print(classification_report(topics_test, topics_pred))

