import csv
import matplotlib.pyplot as plt
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
    all_rows = sum([summ[label]][0][2] for label in summ)
    probs = dict()
    for res, class_summ in summ.items():
        probs[res] = summ[res][0][2]/float(all_rows)
        for i in range(len(class_summ)):
            mean, dev, count = class_summ[i]
            probs[res] *= GaussProbabilityCalc(row[i], mean, dev)
    return probs


if __name__ == '__main__':
    #plt.figure(figsize=(20, 8))
    df = pd.read_csv("./breast-cancer_cleaned_data.csv", delimiter=";", header=1).to_numpy()
    # print(df.dtypes)
    #for c in df :
    #    print(df[c].value_counts(dropna=False))
    #print(summarize_dataset(df))
    summary = SummByClass(df)
    probs = CalsClasProb(summary, df[0])
    print(probs)



    #print(df.head())
    #hm = sns.heatmap(df.corr(), annot=True)
    #hm.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    #plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


