"""
CS795/895
HW2
@author: yu zhang
"""
import pandas as pd
import numpy as np

"""Given two list of numbers that are already sorted, 
 return a single merged list in sorted order.
"""
def merge(sortedListA, sortedListB):
    #Complete this part
    sortedList= sorted(sortedListA + sortedListB)
    return sortedList

"""Given a list of numbers in random order, return the summary statistics 
that includes the max, min, mean, population standard deviation, median,
75 percentile, and 25 percentile.
"""    
def summaryStatistics(listOfNums):
    #Complete this part. 
    # You can decide to return the following statistics either in a sequence 
    # type (i.e., list, tuple), or a key-value pair (i.e., dictionary)
    maxVal = max(listOfNums)
    minVal = min(listOfNums)
    meanVal =  np.mean(listOfNums)
    stdev = np.std(listOfNums)
    median= np.median(listOfNums)
    perc75= np.percentile(listOfNums, 75)
    perc25= np.percentile(listOfNums, 25)
    return {'max': maxVal, 
            'min': minVal, 
            'mean': meanVal, 
            'stdev': stdev,
            'median': median,
            'perc75': perc75,
            'perc25': perc25}
    

"""Given a list of real numbers in any range, scale them to be 
between 0 and 1 (inclusive). For each number x in the list, the new number 
is computed with the formula ((x - min)/(max - min)) where max is the 
maximum value of the original list and min is the minimum value of the list. 
"""	

def scaleToDigits(listOfNums): 
    #complete this part
    newList = [((x-min(listOfNums) )/(max(listOfNums)-min(listOfNums))) for x in listOfNums]
    return newList

#easier way
from sklearn import preprocessing
scaler=preprocessing.MinMaxScaler()
def scaleToDigits2(listOfNums):
    newList= scaler.fit_transform(np.array(listOfNums).reshape(-1,1))

    return newList

list1=[9,19,20,31,41]
list2=[10,8,5]
#print(merge(list1,list2))
