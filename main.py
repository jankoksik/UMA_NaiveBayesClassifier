import itertools
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math
import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
import sklearn as sklearn


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

def GetColumnValues(matrix, i):
    return [row[i] for row in matrix]

class ConfusionMatrix:
    def __init__(self, TP, FN, FP, TN):
        self.TP = TP
        self.FN = FN
        self.FP = FP
        self.TN = TN



    def AddTP(self):
        self.TP+=1
    def AddFN(self):
        self.FN+=1
    def AddFP(self):
        self.FP+=1
    def AddTN(self):
        self.TN+=1
    def __str__(self):
        return "[ " + str(self.TP) + ", " + str(self.FP) + "] \n[" + str(self.FN) + ", " + str(self.TN) + "]\n"

    def CalculateSensitivity(self):
        return self.TP / (self.TP + self.FN)

    def CalculateSpecificity(self):
        return self.TN / (self.FP + self.TN)

    def CalculateAccuracy(self):
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    def CalculatePositivePredictiveValue(self):
        return self.TP / (self.TP + self.FP)

    def CalculateYoudenIndex(self):
        return self.CalculateSensitivity() + self.CalculateSpecificity() - 1

    def CalculateDiscriminationPower(self):
        return math.sqrt(3)/math.pi * (math.log(self.CalculateSensitivity()/ (1-self.CalculateSpecificity()))) + math.log(self.CalculateSpecificity() / (1-self.CalculateSensitivity()))

    def OurOwnScore(self):
        return self.TP + self.TN - 3*self.FN - self.FP






if __name__ == '__main__':

    df = pd.read_csv("./breast-cancer_cleaned_data.csv", delimiter=";", header=1).to_numpy()

    #Confusion Matrix
    ConfMatrixGNBC:ConfusionMatrix = ConfusionMatrix(0,0,0,0)
    ConfMatrixLR:ConfusionMatrix = ConfusionMatrix(0,0,0,0)
    ConfMatrixKN: ConfusionMatrix = ConfusionMatrix(0, 0, 0, 0)
    ConfMatrixDT: ConfusionMatrix = ConfusionMatrix(0, 0, 0, 0)
    ConfMatrixSVM: ConfusionMatrix = ConfusionMatrix(0, 0, 0, 0)
    ConfMatrixRF: ConfusionMatrix = ConfusionMatrix(0, 0, 0, 0)

    #Startify k-fold cross validation
    #12 krotne losowanie
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



        # Spliting data for testing and teaching
        ArrayTeaching:list = list(itertools.chain(*SubSets[0:10]))
        ArrayTest:list = SubSets[11]
        sliceTeachX = [ArrayTeaching[i][1:] for i in range(0, len(ArrayTeaching))]
        sliceTestX = [ArrayTest[i][1:] for i in range(0, len(ArrayTest))]
        sliceTestY = [ArrayTest[i][0:1] for i in range(0, len(ArrayTest))]


        #Gausian Naive Bayes Clasifier
        summary = SummByClass(np.array(ArrayTeaching))
        AllResultsGNGC:list = []
        for n in range(0,len(np.array(ArrayTest))):
            probs = CalsClasProb(summary, ArrayTest[n])
            AllResultsGNGC.append(CheckClass(probs))
            if(CheckClass(probs) == np.array(ArrayTest)[n][0]) :
                if CheckClass(probs) == 0:
                    ConfMatrixGNBC.AddTN()
                if CheckClass(probs) == 1:
                    ConfMatrixGNBC.AddTP()
            else :
                if CheckClass(probs) == 0:
                    ConfMatrixGNBC.AddFN()
                if CheckClass(probs) == 1:
                    ConfMatrixGNBC.AddFP()
        FprGNGC, TprGNGC, thresholds = roc_curve(sliceTestY, AllResultsGNGC, pos_label=1)
        roc_aucGNGC = auc(FprGNGC, TprGNGC)

        # Logic Regression
        logreg = LogisticRegression()
        logreg.fit(sliceTeachX, GetColumnValues(ArrayTeaching, 0))
        LR_ResultPrediction = logreg.predict(sliceTestX)
        FprLR, TprLR, thresholds  = roc_curve(sliceTestY, LR_ResultPrediction, pos_label=1)
        roc_aucLR = auc( FprLR, TprLR)
        for i in range(0, len(LR_ResultPrediction)) :
            if LR_ResultPrediction[i] == sliceTestY[i] :
                if LR_ResultPrediction[i] == 0:
                    ConfMatrixLR.AddTN()
                if LR_ResultPrediction[i] == 1:
                    ConfMatrixLR.AddTP()
            else:
                if LR_ResultPrediction[i] == 0:
                    ConfMatrixLR.AddFN()
                if LR_ResultPrediction[i] == 1:
                    ConfMatrixLR.AddFP()

        # KNearest Neighbour
        KnNeigh = KNeighborsClassifier()
        KnNeigh.fit(sliceTeachX, GetColumnValues(ArrayTeaching, 0))
        KN_ResultPrediction = KnNeigh.predict(sliceTestX)
        FprKN, TprKN, thresholds = roc_curve(sliceTestY, KN_ResultPrediction, pos_label=1)
        roc_aucKN = auc(FprKN, TprKN)
        for i in range(0, len(KN_ResultPrediction)):
            if KN_ResultPrediction[i] == sliceTestY[i]:
                if KN_ResultPrediction[i] == 0:
                    ConfMatrixKN.AddTN()
                if KN_ResultPrediction[i] == 1:
                    ConfMatrixKN.AddTP()
            else:
                if KN_ResultPrediction[i] == 0:
                    ConfMatrixKN.AddFN()
                if KN_ResultPrediction[i] == 1:
                    ConfMatrixKN.AddFP()

        # Decision Tree Classifier
        DecTree = DecisionTreeClassifier()
        DecTree.fit(sliceTeachX, GetColumnValues(ArrayTeaching, 0))
        DT_ResultPrediction = DecTree.predict(sliceTestX)
        FprDT, TprDT, thresholds = roc_curve(sliceTestY, DT_ResultPrediction, pos_label=1)
        roc_aucDT = auc(FprDT, TprDT)
        for i in range(0, len(DT_ResultPrediction)):
            if DT_ResultPrediction[i] == sliceTestY[i]:
                if DT_ResultPrediction[i] == 0:
                    ConfMatrixDT.AddTN()
                if DT_ResultPrediction[i] == 1:
                    ConfMatrixDT.AddTP()
            else:
                if DT_ResultPrediction[i] == 0:
                    ConfMatrixDT.AddFN()
                if DT_ResultPrediction[i] == 1:
                    ConfMatrixDT.AddFP()

        # Support Vector Machine Model
        svc = SVC()
        svc.fit(sliceTeachX, GetColumnValues(ArrayTeaching, 0))
        SVC_ResultPrediction = svc.predict(sliceTestX)
        FprSVC, TprSVC, thresholds = roc_curve(sliceTestY, SVC_ResultPrediction, pos_label=1)
        roc_aucSVC = auc(FprSVC, TprSVC)
        for i in range(0, len(SVC_ResultPrediction)):
            if SVC_ResultPrediction[i] == sliceTestY[i]:
                if SVC_ResultPrediction[i] == 0:
                    ConfMatrixSVM.AddTN()
                if SVC_ResultPrediction[i] == 1:
                    ConfMatrixSVM.AddTP()
            else:
                if SVC_ResultPrediction[i] == 0:
                    ConfMatrixSVM.AddFN()
                if SVC_ResultPrediction[i] == 1:
                    ConfMatrixSVM.AddFP()

        # Random Forest Classifier
        RFC = RandomForestClassifier()
        RFC.fit(sliceTeachX, GetColumnValues(ArrayTeaching, 0))
        RFC_ResultPrediction = RFC.predict(sliceTestX)
        FprRFC, TprRFC, thresholds = roc_curve(sliceTestY, RFC_ResultPrediction, pos_label=1)
        roc_aucRFC = auc(FprRFC, TprRFC)
        for i in range(0, len(RFC_ResultPrediction)):
            if RFC_ResultPrediction[i] == sliceTestY[i]:
                if RFC_ResultPrediction[i] == 0:
                    ConfMatrixRF.AddTN()
                if RFC_ResultPrediction[i] == 1:
                    ConfMatrixRF.AddTP()
            else:
                if RFC_ResultPrediction[i] == 0:
                    ConfMatrixRF.AddFN()
                if RFC_ResultPrediction[i] == 1:
                    ConfMatrixRF.AddFP()


    #print(ConfMatrixGNBC)
    #print(ConfMatrixLR)
    print("Sensitivity :")
    print("GNBC score : " + str(round(ConfMatrixGNBC.CalculateSensitivity() * 100, 2)) + "%")
    print(" LR  score : " + str(round(ConfMatrixLR.CalculateSensitivity()* 100, 2)) + "%")
    print(" KN  score : " + str(round(ConfMatrixKN.CalculateSensitivity()* 100, 2)) + "%")
    print(" DT  score : " + str(round(ConfMatrixDT.CalculateSensitivity()* 100, 2)) + "%")
    print("SVC  score : " + str(round(ConfMatrixSVM.CalculateSensitivity()* 100, 2)) + "%")
    print(" RF  score : " + str(round(ConfMatrixRF.CalculateSensitivity()* 100, 2)) + "%")
    print("\nSpecifity :")
    print("GNBC score : " +  str(round(ConfMatrixGNBC.CalculateSpecificity() * 100, 2)) + "%")
    print(" LR  score : " +  str(round(ConfMatrixLR.CalculateSpecificity() * 100, 2)) + "%")
    print(" KN  score : " +  str(round(ConfMatrixKN.CalculateSpecificity() * 100, 2)) + "%")
    print(" DT  score : " +  str(round(ConfMatrixDT.CalculateSpecificity() * 100, 2)) + "%")
    print("SVC  score : " + str(round(ConfMatrixSVM.CalculateSpecificity() * 100, 2)) + "%")
    print(" RF  score : " +  str(round(ConfMatrixRF.CalculateSpecificity() * 100, 2)) + "%")
    print("\nAccuracy :")
    print("GNBC Acc : " + str(round(ConfMatrixGNBC.CalculateAccuracy() * 100, 2)) + "%")
    print(" LR  Acc : " + str(round(ConfMatrixLR.CalculateAccuracy() * 100,2)) + "%")
    print(" KN  Acc : " + str(round(ConfMatrixKN.CalculateAccuracy() * 100,2)) + "%")
    print(" DT  Acc : " + str(round(ConfMatrixDT.CalculateAccuracy() * 100,2)) + "%")
    print("SVC  Acc : " + str(round(ConfMatrixSVM.CalculateAccuracy() * 100,2)) + "%")
    print(" RF  Acc : " + str(round(ConfMatrixRF.CalculateAccuracy() * 100,2)) + "%")
    print("\nYounden's index :")
    print("GNBC score : " + str(round(ConfMatrixGNBC.CalculateYoudenIndex()* 100, 2)) + "%")
    print(" LR  score : " + str(round(ConfMatrixLR.CalculateYoudenIndex()* 100, 2)) + "%")
    print(" KN  score : " + str(round(ConfMatrixKN.CalculateYoudenIndex()* 100, 2)) + "%")
    print(" DT  score : " + str(round(ConfMatrixDT.CalculateYoudenIndex()* 100, 2)) + "%")
    print("SVC  score : " + str(round(ConfMatrixSVM.CalculateYoudenIndex()* 100, 2)) + "%")
    print(" RF  score : " + str(round(ConfMatrixRF.CalculateYoudenIndex()* 100, 2)) + "%")
    print("\nPositive predictive value :")
    print("GNBC score : " + str(round(ConfMatrixGNBC.CalculatePositivePredictiveValue()* 100, 2)) + "%")
    print(" LR  score : " + str(round(ConfMatrixLR.CalculatePositivePredictiveValue()* 100, 2)) + "%")
    print(" KN  score : " + str(round(ConfMatrixKN.CalculatePositivePredictiveValue()* 100, 2)) + "%")
    print(" DT  score : " + str(round(ConfMatrixDT.CalculatePositivePredictiveValue()* 100, 2)) + "%")
    print("SVC  score : " + str(round(ConfMatrixSVM.CalculatePositivePredictiveValue()* 100, 2)) + "%")
    print(" RF  score : " + str(round(ConfMatrixRF.CalculatePositivePredictiveValue()* 100, 2)) + "%")
    print("\nDiscrimination power :")
    print("GNBC score : " + str(ConfMatrixGNBC.CalculateDiscriminationPower()))
    print(" LR  score : " + str(ConfMatrixLR.CalculateDiscriminationPower()))
    print(" KN  score : " + str(ConfMatrixKN.CalculateDiscriminationPower()))
    print(" DT  score : " + str(ConfMatrixDT.CalculateDiscriminationPower()))
    print("SVC  score : " + str(ConfMatrixSVM.CalculateDiscriminationPower()))
    print(" RF  score : " + str(ConfMatrixRF.CalculateDiscriminationPower()))
    print("\nOur Score :")
    print("GNBC score : " + str(ConfMatrixGNBC.OurOwnScore()))
    print(" LR  score : " + str(ConfMatrixLR.OurOwnScore()))
    print(" KN  score : " + str(ConfMatrixKN.OurOwnScore()))
    print(" DT  score : " + str(ConfMatrixDT.OurOwnScore()))
    print("SVC  score : " + str(ConfMatrixSVM.OurOwnScore()))
    print(" RF  score : " + str(ConfMatrixRF.OurOwnScore()))


    fig, ax = plt.subplots(figsize=(10, 10))

    plt.plot(FprGNGC, TprGNGC, label='ROC Curve GNBC (AUC = %0.2f)' % (roc_aucGNGC), alpha=0.7)
    plt.plot(FprLR, TprLR, label='ROC Curve LRC (AUC = %0.2f)' % (roc_aucLR), alpha=0.7)
    plt.plot(FprKN, TprKN, label='ROC Curve KNC (AUC = %0.2f)' % (roc_aucKN), alpha=0.7)
    plt.plot(FprDT, TprDT, label='ROC Curve DTC (AUC = %0.2f)' % (roc_aucDT), alpha=0.7)
    plt.plot(FprSVC, TprSVC, label='ROC Curve SVC (AUC = %0.2f)' % (roc_aucSVC), alpha=0.7)
    plt.plot(FprRFC, TprRFC, label='ROC Curve RFC (AUC = %0.2f)' % (roc_aucRFC), alpha=0.7)

    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.show()












