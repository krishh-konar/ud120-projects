#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test) 

# print 'num training pts: ' + str(len(labels_train)) +' :; ' + str(len(features_train[0])) + ' :; ' + str(len(pred))
# print accuracy_score(labels_test,pred)
# print len(clf.feature_importances_)
features = clf.feature_importances_


# What's the importance of the most important feature? What is the number of this feature? 
# loop_index, count,most_imp,index = 0,0,0,0
# for f in features:
# 	if f>0.2:
# 		count+=1
# 		if f>most_imp:
# 			most_imp=f
# 			index=loop_index
# 	loop_index+=1

# print most_imp,index

# What is it? Does it make sense as a word that's uniquely tied to either Chris Germany or Sara Shackleton, a signature of sorts?
tfidf_list = vectorizer.get_feature_names()
# print tfidf_list[33614]


#Any other outliers pop up? What word is it? Seem like a signature-type word? 
loop_index, count,most_imp,index = 0,0,0,[]
for f in features:
	if f>0.2:
		print f
		index.append(loop_index)
	loop_index+=1

print index

outliers = []

for i in index:
	outliers.append(tfidf_list[i])

print outliers

#final accuracy score
print accuracy_score(labels_test,pred)