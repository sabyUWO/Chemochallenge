from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import KFold
import numpy as np
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import pandas as pd
import numpy as np
import sklearn as sk
import csv
import pandas as pd
import csv
from rdkit.Chem import AllChem
from sklearn import preprocessing
#import matplotlib.pyplot as plt
#plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
import boltzmannclean
import os
import pandas
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


pd.set_option('mode.chained_assignment', None)
pd.options.mode.chained_assignment = None


def ExplicitBitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def fillMissingData(incompleteDf):

    completeDf = pd.DataFrame(np.zeros(incompleteDf.shape))

    completeDf = boltzmannclean.clean(
        dataframe=incompleteDf,
        numerical_columns=['target1', 'target2', 'target3', 'target4', 'target5', 'target6', 'target7', 'target8',
                           'target9', 'target10', 'target11', 'target12'],
        categorical_columns=[],
        tune_rbm=True  # tune RBM hyperparameters for my data
    )

    completeDf[completeDf > 0.5] = 1
    completeDf[completeDf < 0.5] = 0

    return completeDf

userhome = os.path.expanduser('~')
csvfile= userhome + r'/Documents/cheminf/data/data.csv' # Change here
print(csvfile)
with open(csvfile, "r") as f:
    df = pandas.read_csv(csvfile)

#df = df[1:200]
print('df.shape',df.shape)
#df_sub_smiles = df
#df = df.drop(columns=['smiles'])
#print('df',df.columns)
#df.sub = df
print('list(df)',list(df))


kf = KFold(n_splits=10)
kf.get_n_splits(df)
keydict ={
  1: "target1",   2: "target2",   3: "target3",
  4: "target4",   5: "target5",   6: "target6",
  7: "target7",   8: "target8",   9: "target9",
  10: "target10",  11: "target11",   12: "target12"
}
for targetNum in range(1, 13):
    auclist = []
    accuracylist = []
    precisionlist = []
    recallList = []
    f1List  = []

    for train_index, test_index in kf.split(df):

        train = df.iloc[train_index]
        train = train.reset_index(drop=True)
        test = df.iloc[test_index]
        test = test.reset_index(drop=True)

        trainTargets = df.iloc[train_index,1:13]
        trainTargets = trainTargets.reset_index(drop=True)

        #print('list(trainTargets)',list(trainTargets))
        trainTargets = fillMissingData(trainTargets)

        #print('trainTargets',trainTargets)
        #print(' i is .... ', targetNum)
        #print('keydict[i] 1',keydict[targetNum])
        trainY = trainTargets.loc[:,trainTargets.columns == keydict[targetNum]]

        #print('trainY',trainY)

        trainTargetNotY = trainTargets.loc[:, trainTargets.columns != keydict[targetNum]]

        testTargets = df.iloc[test_index,1:13]
        testTargets = testTargets.reset_index(drop=True)

        #print('list(testTargets)',list(testTargets))
        testTargets = fillMissingData(testTargets)
        #print('keydict[i] 2',keydict[targetNum])
        testY = testTargets.loc[:,testTargets.columns == keydict[targetNum]]
        testTargetNotY = testTargets.loc[:, testTargets.columns != keydict[targetNum]]

        #print('len(train_index)',len(train_index))
        #print('len(test_index)',len(test_index))
        trainX = pd.DataFrame(np.zeros((len(train_index), 177)))
        testX = pd.DataFrame(np.zeros((len(test_index), 177)))

        for i, j in train.iterrows():
            fingerprint = ExplicitBitVect_to_NumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(j['smiles'])))[1:]
            #print('i 1',i)
            trainX.iloc[i] = np.append(fingerprint, trainTargetNotY.iloc[i])

        for i, j in test.iterrows():
            fingerprint = ExplicitBitVect_to_NumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(j['smiles'])))[1:]
            #print('i 2',i)
            testX.iloc[i] = np.append(fingerprint, testTargetNotY.iloc[i])


        #print('np.sum(trainY)',np.sum(trainY[keydict[targetNum]]))
        #print('trainY',trainY[keydict[targetNum]])
        #print('testY',testY[keydict[targetNum]])
        tr = np.sum(trainY[keydict[targetNum]])
        ts = np.sum(testY[keydict[targetNum]])
        if (tr > 0 and ts > 0):
            model = LogisticRegression(random_state=0, solver='lbfgs',multi_class = 'multinomial').fit(trainX, trainY)
            predictY = model.predict(testX)
            predictY = predictY.reshape((predictY.shape[0], 1))
            #print('predictY.shape',predictY.shape)
            #print('testY.shape',testY.shape)
            accuracy = np.mean(predictY == testY)
            accuracylist.append(accuracy)
            #print('accuracy',accuracy)
            #try:
            auc = roc_auc_score(testY, predictY)
            auclist.append(auc)
            #print('auc',auc)
            precision = precision_score(testY, predictY)
            #print ('precsion',precision)
            recall = recall_score(testY, predictY)
            #print ('recall',recall)
            f1 = f1_score(testY, predictY)
            #print ('f1',f1)

            f1List.append(f1)
            precisionlist.append(precision)
            recallList.append(recall)
            #except :
            #pass

    print('Target ',targetNum,' AUC ',np.mean(auclist),' Accuracy ',np.mean(accuracylist),' Precision ', np.mean(precisionlist), ' Recall ', np.mean(recallList), 'F1 score' , np.mean(f1List))

