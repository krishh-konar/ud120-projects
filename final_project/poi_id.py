#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas
import sklearn

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import matplotlib.pyplot as pl


###############################################
### Task 1: Select what features you'll use ###
###############################################

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
				 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'long_term_incentive', 
                 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# print len(data_dict)
# print len(data_dict["METTS MARK"])

data_frame = pandas.DataFrame.from_records(list(data_dict.values()))
person = pandas.Series(list(data_dict.keys()))

#print data_frame.head()
data_frame.replace(to_replace='NaN', value=np.nan, inplace=True)
#print data_frame.isnull().sum()

#remove features which have more than 70 NaN values (50%)
for column, series in data_frame.iteritems():
	if series.isnull().sum() >=70:
		data_frame.drop(column, axis=1, inplace=True)

#email gives no real infortmation about poi, drop it
if 'email_address' in list(data_frame.columns.values):
    data_frame.drop('email_address', axis=1, inplace=True)

data_frame = data_frame.replace(to_replace=np.nan, value=0)


###############################
### Task 2: Remove outliers ###
###############################

data_dict.pop("YEAP SOON")
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")


#####################################
### Task 3: Create new feature(s) ###
#####################################


### An importtant feature could be the fraction of emails sent to/from poi to the total number of emails sent, as POIs are more
### more likely to contact other POIs than non POIs.

#total ratio
poi_msg_ratio = ( data_frame['from_this_person_to_poi'] + data_frame['from_poi_to_this_person'] ) \
				 / (data_frame['to_messages'] + data_frame['from_messages'])

#send ratio
poi_send_ratio = data_frame['from_this_person_to_poi'] / data_frame['from_messages']

#received ratio
poi_recieved_ratio = data_frame['from_poi_to_this_person'] / data_frame['to_messages']

##Scaling salary and bonuses to a scale of (0-100) for better visualization

#defing scale
scale = sklearn.preprocessing.MinMaxScaler(feature_range=(0,100), copy= True)

#scaled features
scaled_salary = scale.fit_transform(np.array(data_frame['salary']).reshape(len(data_frame['salary']),1))
scaled_bonus = scale.fit_transform(np.array(data_frame['bonus']).reshape(len(data_frame['bonus']),1))

#adding features to DataFrame
data_frame['poi_msg_ratio'] = pandas.Series(poi_msg_ratio) * 100
data_frame['poi_send_ratio'] = pandas.Series(poi_send_ratio) * 100
data_frame['poi_recieved_ratio'] = pandas.Series(poi_recieved_ratio) *100

#print data_frame.describe()

# pl.scatter(x=data_frame['salary'],y=data_frame['poi_send_ratio'],c=data_frame['poi'])
# pl.show()


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


###########################################
### Task 4: Try a varity of classifiers ###
###########################################

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


cv = sklearn.cross_validation.StratifiedShuffleSplit(labels, n_iter=10)

#clf_RF = RandomForestClassifier()
#clf_ADB = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9))
#clf_KNN = KNeighborsClassifier()
	 	
# scores1 = sklearn.cross_validation.cross_val_score(clf_RF, features, labels)
# print scores1
# scores2 = sklearn.cross_validation.cross_val_score(clf_ADB, features, labels)
# print scores2
# scores3 = sklearn.cross_validation.cross_val_score(clf_KNN, features, labels)
# print scores3

##############################################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
###############################################################################################################

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)


import time

cols = list(data_frame.columns.values)
cols.remove('poi')
cols.insert(0, 'poi')
data = data_frame[cols].fillna(0).to_dict(orient='records')
enron_data_sub = {}
counter = 0
for item in data:
    enron_data_sub[counter] = item
    counter += 1



### Random FOrest Classifier ###
################################

# start_time = time.time()
# params = {"max_depth":[2,3,4,5,6], 'min_samples_split':[2,3,4,5,6], 'n_estimators':[10,20,40],
#  'min_samples_leaf':[1,2,3], 'criterion':('gini', 'entropy')}

# cv_RF = GridSearchCV(clf_RF, params)
# cv_RF.fit(features,labels)
# clf1 = cv_RF.best_estimator_
# print cv_RF.best_score_

# test_classifier(clf1,enron_data_sub,cols)
# elapsed_time= start_time - time.time()
# print -elapsed_time
# print


### AdaBoost Classifier ###
###########################

start_time = time.time()

params = {'kbest__k': range(13,16), 'ADB__n_estimators':[10,20,30,100,150,200],'ADB__learning_rate':[1.0,1.5,2],
			"ADB__base_estimator__criterion" : ["gini"] }

DTC = DecisionTreeClassifier(max_depth = 8)
ABC = AdaBoostClassifier(base_estimator = DTC)

kbest = SelectKBest(f_classif)

pipeline = Pipeline([('kbest', kbest), ('ADB', ABC)])
grid_search = GridSearchCV(pipeline, param_grid=params)
grid_search.fit(features, labels)

clf1 = grid_search.best_estimator_

kbest.fit_transform(features,labels)
# print data_frame.columns
# print kbest.scores_

test_classifier(clf1,enron_data_sub,cols)
elapsed_time= start_time - time.time()
print -elapsed_time
print



### KNN Classifier ###
#######################

# start_time = time.time()

# params = {'n_neighbors': [3,4,5,6] ,  'weights':['uniform','distance'],'leaf_size':[15,20,25,30,40], 'n_jobs':[-1]}
# cv_KNN = GridSearchCV(clf_KNN, params)
# cv_KNN.fit(features, labels)
# clf1 = cv_KNN.best_estimator_
# print cv_KNN.best_score_

# #test_classifier(clf1,my_dataset,features_list)
# test_classifier(clf1,enron_data_sub,cols)

# elapsed_time= start_time - time.time()
# print elapsed_time
# print


###################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
###################################################################################

dump_classifier_and_data(clf1, enron_data_sub, cols)
