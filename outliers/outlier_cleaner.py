#!/usr/bin/python

from sklearn.linear_model import LinearRegression

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    uncleaned_data = []

    for pred, age, net_worth in zip(predictions, ages, net_worths):
        error = (pred[0] - net_worth[0])**2
        print age[0],net_worth[0],error
        uncleaned_data.append((age[0],net_worth[0],float(error)))

    #print 'reached 1\n'
    #print uncleaned_data
    #print 'reached 2\n'

    uncleaned_data.sort(key = lambda val:val[2])

    #print uncleaned_data
    #print '\n', len(uncleaned_data)
    #print 'reached 3\n'

    uncleaned_data = uncleaned_data[:-(len(uncleaned_data)/10)]
    #print 'reached 4\n'
    #print uncleaned_data
    #print '\n', len(uncleaned_data)

    return uncleaned_data

