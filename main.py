import collections
import csv
import itertools
import random

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
import numpy as np
import math


# recurrence-events - 1 | no-recurrence-events - 0
# 10-19 - 1 | 0-29 -2 | 30-39 - 3 | 40-49 - 4 | 50-59 - 5 |  60-69 - 6 | 70-79 - 7 | 80-89 - 8|  90-99 -9
# premeno - 1 | ge40 - 2 | lt40 - 3
# 0-4 - 1, 5-9 -2, 10-14 -3, 15-19 -4, 20-24 -5, 25-29 -6, 30-34 -7, 35-39 -8, 40-44 -9, 45-49 -10, 50-54 -11, 55-59 -12
# 0-2 -1 , 3-5 -2, 6-8 -3, 9-11 -4, 12-14 -5, 15-17 -6, 18-20 -7, 21-23 -8, 24-26 -9, 27-29 -10 , 30-32 -11, 33-35 -12, 36-39 -13
#  yes -1, no -0
# left -1, right -2
# left-up -1, left-low -2,  right-up-3, right-low-4, central-5.
# yes -1, no -0

# returns dictionary
# splits records by values (has cancer or no)
def SeperateByResult(ds):
    seperated = dict()
    for i in range(len(ds)):
        vector = ds[i]
        result = vector[0]
        if result not in seperated :
            seperated[result] = list()
        seperated[result].append(vector)
    return seperated

#calculates mean from list of values
def mean(nmbr):
    return sum(nmbr)/float(len(nmbr))

def StandardDeviation(nmbr):
    avg = mean(nmbr)
    variance = sum([(x - avg) ** 2 for x in nmbr]) / float(len(nmbr) - 1)
    return math.sqrt(variance)

def summarize_dataset(ds):
    summaries = [(mean(column), StandardDeviation(column), len(column)) for column in zip(*ds)]
    del(summaries[-1])
    return summaries

def SummByClass(ds):
    sep = SeperateByResult(ds)
    summ = dict()
    for res,rows in sep.items():
        summ[res] = summarize_dataset(rows)
    return summ

def GaussProbabilityCalc(x, mean, dev):
    ex = math.exp(-((x-mean)**2 / (2 * dev**2)))
    return (1 / (math.sqrt(2*math.pi) * dev )) * ex

def CalsClasProb(summ, row):
    all_rows = sum([summ[label][0][2] for label in summ])
    probs = dict()
    for res, class_summ in summ.items():
        probs[res] = summ[res][0][2]/float(all_rows)
        for i in range(len(class_summ)):
            mean, dev, count = class_summ[i]
            if dev == 0:
                continue
            probs[res] *= GaussProbabilityCalc(row[i], mean, dev)
    return probs

def CheckClass(probs):
    if probs[0] > probs[1]:
        return 0
    else :
        return 1

def CalcNodeCaps(subset):
    counter = 0
    for s in subset :
        if s[5] == 0 :
            counter+=1
    return counter


if __name__ == '__main__':

    df = pd.read_csv("./breast-cancer_cleaned_data.csv", delimiter=";", header=1).to_numpy()

    #Startify k-fold cross validation
    #12 krotne losowanie
    AccAll:float = 0
    for x in range(0,12):
        K = 12
        SubSets:list = []
        HowManyElements = int(len(df)/K)
        ncz:list = []
        nco:list = []

        for n in df :
            if n[5]==0:
                ncz.append(n)
            else :
               nco.append(n)




        for ssf in range(0,K-1):
            SubSet:list = []
            for ssvz in range(0,18):
                    index = random.choice(range(len(ncz)))
                    SubSet.append(ncz[index-1])
                    ncz.pop(index)
            for ssvo in range(0,5):
                index = random.choice(range(len(nco)))
                SubSet.append(nco[index-1])
                nco.pop(index)
            SubSets.append(SubSet)

        SubSets.append(nco+ncz)




        ArrayTeaching:list = list(itertools.chain(*SubSets[0:10]))
        ArrayTest:list = SubSets[11]
        summary = SummByClass(np.array(ArrayTeaching))
        OK:int = 0
        AtALL:int = 0
        for n in range(0,len(np.array(ArrayTest))):
            probs = CalsClasProb(summary, ArrayTest[n])
            if(CheckClass(probs) == np.array(ArrayTest)[n][0]) :
                OK +=1;
            AtALL+=1;

        AccAll += OK/AtALL;

    AccAll /= 12
    print("Accuracy : " + str(AccAll*100))









