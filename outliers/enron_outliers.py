#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression 
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL",None)
data = featureFormat(data_dict, features)

print type(data_dict), type(data)
reg = LinearRegression()




#reg.fit(data_dict["salary"],data_dict["bonus"])
cnt=0
for i in data_dict:
	if data_dict[i]["bonus"] > 5000000 and data_dict[i]["salary"] > 1000000 and data_dict[i]["bonus"] != "NaN" and data_dict[i]["salary"] != "NaN":
		cnt+=1
		print i
print cnt

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

#plt.plot(salary, reg.predict(salary), color="blue")
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()