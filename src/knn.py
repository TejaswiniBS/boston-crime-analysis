import numpy as np
import pandas as pd
from pyspark.sql.functions import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding

import matplotlib.pyplot as plt
# For WSL2, set the DISPLAY variable(run Mobaxterm)
matplotlib.use('tkagg')

def knn(data_dir):
    #***************************************************************************************************
    # Output
    # Test accuracy: 0.45916697531903433
    # Accuracy: 0.2610109834854889
    # F1 score: 0.2280568612454126

    #***************************************************************************************************

    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    # space1_list = ['Lat','Long'] 
    # space2_list = ['DISTRICT','REPORTING_AREA','STREET']
    # time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR']
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.44446894663802444, F1 score: 0.42777359422852407


    # features_list = all
    # prediction_type='OFFENSE_CODE_GROUP'
    #*************************************************************************************************************************************

    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    space1_list = ['Lat','Long'] # Accuracy = 0.9954459665969783 F1 score: 0.9954511025542934
    space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.5412653440247267  F1 score: 0.48845570294558494
    time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.8542675247913225 F1 score: 0.8507342536482208
    time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =0.8559382099620931  F1 score: 0.8525141315197746
    space1_time_list = space1_list + time1_list # Accuracy = 0.8490561562467175 F1 score: 0.845004598760486
    space2_time_list = space2_list + time1_list # Accuracy = 0.530691438525618 F1 score: 0.4746146384288717
    all = space1_list  + space2_list + time1_list # Accuracy =  0.5323302113721275  F1 score: 0.4796943788144287

    # # # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.6468094328633156  F1 score: 0.5994901951808032

    # # # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.6299426976760015  F1 score: 0.5810812869437284

    features_list = all + ['OFFENSE_CODE_GROUP']
    prediction_type='UCR_PART'
    #*************************************************************************************************************************************

    # Cleanse Data
    df = cleanse_data(data_dir=data_dir, features_list=features_list, prediction_type=prediction_type)
    df = encoding(df, features_list)


    data=df.toPandas()
    X = data.loc[:, data.columns != 'label']
    Y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9) # 90% training and 10% test

    error_rate = []
    krange = range(20,100,5)
    krange = [40]
    for i in krange:
        knn = KNeighborsClassifier(n_neighbors=i) # metric='manhattan', weights = 'uniform',n_jobs= -1 # (distance metric)haversine - if only lat long
        knn.fit(X_train,y_train)
        pred_train = knn.predict(X_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
        # f'Test accuracy: {metrics.accuracy_score(y_train, pred_train)} \n'
        print(f'Neighbours: {i}\n'
            f'Accuracy: {metrics.accuracy_score(y_test, pred_i)}\n'
            f'F1 score: {metrics.f1_score(y_test, pred_i, average = "weighted")}')

        unique_label = np.unique([y_test, pred_i])
        cmtx = pd.DataFrame(
        confusion_matrix(y_test, pred_i, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
        )
        print(cmtx)

    # plt.figure(figsize=(10,6))
    # plt.plot(krange,error_rate, color= 'blue', linestyle= 'dashed', marker= 'o', markerfacecolor='red', markersize=1)
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')
    # plt.show()

    # optimum K value == 40

    #### Experiment
    # ------------------------------------------------------------------------------------------------
    # Predicting Severity/Major/Minor
    ## ------------------------------------------------------------------------------------------------
    # space1_list = ['Lat','Long'] # Accuracy = 0.45237608932461876, F1 score: 0.4279084524785869
    # space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.44284764911114805, F1 score: 0.4176126098966624
    # time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.42055381572441075, F1 score: 0.3664281380443766
    # time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy = 0.42106262675486056  , F1 score: 0.36911434206584487
    # space1_time_list = space1_list + time1_list # Accuracy = 0.4235695049624788 , F1 score: 0.3681944727284493
    # space2_time_list = space2_list + time1_list # Accuracy = 0.44496594118624355 , F1 score: 0.4201129800424681
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.4423702070224132, F1 score: 0.4187834281982415

    # # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.6267287035908272  F1 score:0.5845422165627002

    # # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.5590737470609469  F1 score: 0.5434126058849054

    # # # # Only Type1, devided into 7 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.6015822948446146  F1 score: 0.4616437486499766

    # # # # Only Type2, devided into 22 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.2431223365090698  F1 score: 0.20395280657589882

    # # # # Only Type 1 & 2, devided into 29 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.2509312311214055 F1 score: 0.18217038768868937

    # features_list = all
    # prediction_type='OFFENSE_CODE_GROUP'
    #*************************************************************************************************************************************


