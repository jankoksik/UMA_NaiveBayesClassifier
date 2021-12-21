import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# recurrence-events - 1 | no-recurrence-events - 0
# 10-19 - 1 | 0-29 -2 | 30-39 - 3 | 40-49 - 4 | 50-59 - 5 |  60-69 - 6 | 70-79 - 7 | 80-89 - 8|  90-99 -9
# premeno - 1 | ge40 - 2 | lt40 - 3
# 0-4 - 1, 5-9 -2, 10-14 -3, 15-19 -4, 20-24 -5, 25-29 -6, 30-34 -7, 35-39 -8, 40-44 -9, 45-49 -10, 50-54 -11, 55-59 -12
# 0-2 -1 , 3-5 -2, 6-8 -3, 9-11 -4, 12-14 -5, 15-17 -6, 18-20 -7, 21-23 -8, 24-26 -9, 27-29 -10 , 30-32 -11, 33-35 -12, 36-39 -13
#  yes -1, no -0
# left -1, right -2
# left-up -1, left-low -2,  right-up-3, right-low-4, central-5.
# yes -1, no -0





if __name__ == '__main__':
    #plt.figure(figsize=(20, 8))
    df = pd.read_csv("./breast-cancer_cleaned_data.csv" , delimiter=",")
    for c in df :
        print(df[c].value_counts(dropna=False))

    #print(df.head())
    #hm = sns.heatmap(df.corr(), annot=True)
    #hm.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    #plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


