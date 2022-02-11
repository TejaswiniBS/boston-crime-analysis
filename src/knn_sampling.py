import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib


import matplotlib.pyplot as plt


from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding

def knn_sampling(data_dir):

    # For WSL2, set the DISPLAY variable(run Mobaxterm)
    matplotlib.use('tkagg')

    #***************************************************************************************************
    # Output
    # Test accuracy: 0.45916697531903433
    # Accuracy: 0.2610109834854889
    # F1 score: 0.2280568612454126

    #***************************************************************************************************

    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    space1_list = ['Lat','Long'] # Accuracy = , F1 score: 
    space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = , F1 score: 
    time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = , F1 score: 
    time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  , F1 score: 
    # space1_time_list = space1_list + time1_list # Accuracy =  , F1 score: 
    # space2_time_list = space2_list + time1_list # Accuracy = , F1 score: 
    all = space1_list  + space2_list + time1_list # Accuracy = , F1 score: 

    # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score:

    # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score: 

    # # # Only Type1, devided into 7 categories
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score: 

    # # # Only Type2, devided into 22 categories
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score: 

    # # # Only Type 1 & 2, devided into 29 categories
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score: 

    features_list = all
    prediction_type='OFFENSE_CODE_GROUP'
    #*************************************************************************************************************************************

    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    # space1_list = ['Lat','Long'] # Accuracy =  F1 score: 
    # space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy =  F1 score: 
    # time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy =  F1 score: 
    # time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  F1 score: 
    # space1_time_list = space1_list + time1_list # Accuracy = F1 score: 
    # space2_time_list = space2_list + time1_list # Accuracy =  F1 score: 
    # all = space1_list  + space2_list + time1_list # Accuracy =   F1 score: 

    # # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score: 

    # # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy =  F1 score: 

    # features_list = all + ['OFFENSE_CODE_GROUP']
    # prediction_type='UCR_PART'
    #*************************************************************************************************************************************

    # Cleanse Data
    df = cleanse_data(data_dir=data_dir, features_list=features_list, prediction_type=prediction_type)
    df = encoding(df, features_list)


    data=df.toPandas()
    X = data.loc[:, data.columns != 'label']
    Y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9) # 90% training and 10% test

    # error_rate = []
    # krange = range(5,100,5)
    # for i in krange:
    #     knn = KNeighborsClassifier(n_neighbors=i) # metric='manhattan', weights = 'uniform',n_jobs= -1 # (distance metric)haversine - if only lat long
    #     knn.fit(X_train,y_train)
    #     pred_train = knn.predict(X_train)
    #     pred_i = knn.predict(X_test)
    #     error_rate.append(np.mean(pred_i != y_test))
    #     print(f'Neighbours: {i}\n'
    #         f'Test accuracy: {metrics.accuracy_score(y_train, pred_train)} \n'
    #         f'Accuracy: {metrics.accuracy_score(y_test, pred_i)}\n'
    #         f'F1 score: {metrics.f1_score(y_test, pred_i, average = "weighted")}')

    # plt.figure(figsize=(10,6))
    # plt.plot(krange,error_rate, color= 'blue', linestyle= 'dashed', marker= 'o', markerfacecolor='red', markersize=1)
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')
    # plt.show()

    # Findout optimum K value == 40
    #---------------------------------------
    # Grid Search/Random Search -KNN
    # KNeighborsClassifier().get_params().keys()
    grid_params = {
        'weights' : ['uniform', 'distance'],
        'metric' : ['manhattan','jaccard'] #haversine - if only lat long
    }

    randomSearch = RandomizedSearchCV(
        KNeighborsClassifier(40), # Seed the optimum value from the previous step
        grid_params,
        verbose = 1,
        cv =3,
        random_state = 123
    )

    rs_results = randomSearch.fit(X_train, y_train)
    rs_results.best_score_
    # 0.2736051882419795 - Manual grouping
    # 0.4499195838894022 - T1/T2/T3


    grideSearch = GridSearchCV(KNeighborsClassifier(40), # Seed the optimum value from the previous step
        grid_params,
        verbose = 1,
        cv =3
    )
    gs_results = grideSearch.fit(X_train, y_train)
    gs_results.best_score_
    # 0.2736051882419795 - MAnual grouping
    # 0.4499195838894022 - T1/T2/T3

    # -------------------------------------------------------------
    # Oversmapling - SMOTE to balance dataset
    # -------------------------------------------------------------
    # Oversample 'Majority'
    sm = SMOTE('minority', random_state=123)

    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    #convert back to Dataframe
    X_train_updated = pd.DataFrame(X_train_res, columns= features_list)
    y_train_updated = pd.Series(y_train_res)

    target_counts = y_train_updated.value_counts()
    target_counts
    knn = KNeighborsClassifier(n_neighbors=40, metric='manhattan', weights = 'uniform')
    knn.fit(X_train_updated,y_train_updated)
    pred_i = knn.predict(X_test)
    # print('Accuracy ', metrics.accuracy_score(y_test, pred_i))
    sm_accuracy = metrics.accuracy_score(y_test, pred_i)
    # Accuracy  0.2711302462249179, 42%
    # print('F1 score ', metrics.f1_score(y_test, pred_i, average = 'weighted'))
    sm_f1_score = metrics.f1_score(y_test, pred_i, average = 'weighted')
    # F1 score  0.22668225021123448

    # -------------------------------------------------------------
    # Undersampling - Cluster Centroids
    # -------------------------------------------------------------
    # Undersample 'Majority'
    cc = ClusterCentroids(sampling_strategy = 'majority')
    X_cc, y_cc = cc.fit_resample(X_train, y_train)
    #convert back to Dataframe
    X_train_updated = pd.DataFrame(X_cc, columns= features_list)
    y_train_updated = pd.Series(y_cc)
    target_counts = y_train_updated.value_counts()
    target_counts
    knn = KNeighborsClassifier(n_neighbors=40, metric='manhattan', weights = 'uniform')
    knn.fit(X_train_updated,y_train_updated)
    pred_i = knn.predict(X_test)

    cc_accuracy = metrics.accuracy_score(y_test, pred_i)
    # print('Accuracy ', metrics.accuracy_score(y_test, pred_i))
    # Accuracy  0.18324664917270314, 27%

    cc_f1_score = metrics.f1_score(y_test, pred_i, average = 'weighted')
    # print('F1 score ', metrics.f1_score(y_test, pred_i, average = 'weighted'))
    # F1 score  0.1366370177686179


    # -------------------------------------------------------------
    # Random Sampling
    # -------------------------------------------------------------
    # define oversampling strategy
    over = RandomOverSampler(sampling_strategy= 'minority') 

    # fit and apply the transform
    X_random, y_random = over.fit_resample(X_train, y_train)
    #convert back to Dataframe
    X_train_updated = pd.DataFrame(X_random, columns= features_list)
    y_train_updated = pd.Series(y_random)
    target_counts_random = y_train_updated.value_counts()
    target_counts_random

    knn = KNeighborsClassifier(n_neighbors=40, metric='manhattan', weights = 'uniform')
    knn.fit(X_train_updated,y_train_updated)
    pred_i = knn.predict(X_test)
    rand_over_samp_acc = metrics.accuracy_score(y_test, pred_i)
    # print('Accuracy ', metrics.accuracy_score(y_test, pred_i))
    # Accuracy  0.2804952782011596, 46.8%
    rand_over_samp_f1score = metrics.f1_score(y_test, pred_i, average = 'weighted')
    # print('F1 score ', metrics.f1_score(y_test, pred_i, average = 'weighted'))
    # F1 score  0.22978759677714766

    # define undersampling strategy
    under = RandomUnderSampler(sampling_strategy= 'majority')
    # fit and apply the transform
    X_train_updated, y_train_updated = under.fit_resample(X_train, y_train)

    y_train_updated = pd.Series(y_train_updated)
    target_counts_random = y_train.value_counts()
    target_counts_random
    knn = KNeighborsClassifier(n_neighbors=40, metric='manhattan', weights = 'uniform')

    knn.fit(X_train_updated,y_train_updated)
    pred_i = knn.predict(X_test)

    rand_under_samp_acc = metrics.accuracy_score(y_test, pred_i)
    # print('Accuracy ', metrics.accuracy_score(y_test, pred_i))
    # Accuracy  0.1824609921277164

    rand_under_samp_f1score = metrics.f1_score(y_test, pred_i, average = 'weighted')
    # print('F1 score ', metrics.f1_score(y_test, pred_i, average = 'weighted'))
    # F1 score  0.1384086249439674

    # -------------------------------------------------------------
    # Ensemble models
    # -------------------------------------------------------------
    #KNN with - 40 neighbours
    knn = KNeighborsClassifier(40, metric='manhattan', weights = 'uniform',n_jobs= -1)

    #random forest with 100 Decision Trees
    rf = RandomForestClassifier(100, max_depth =25, n_jobs= -1)

    #Support Vector Classifier
    svm = SVC(kernel = 'linear', random_state = 123, probability=True)

    # can using previopusly trained KNN as well
    knn.fit(X_train,y_train)
    rf.fit(X_train,y_train)
    # rf.score(X_test, y_test) 
    # 0.2983925456859572, #0.4842947156707154

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) # Use 10% otherwise SVN will take days to complete
    # svm.fit(X_train,y_train)
    # svm.score(X_test, y_test)

    #create a dictionary of our models
    # estimators=[('knn', knn), ('rf', rf), ('svm', svm)]
    estimators=[('knn', knn), ('rf', rf)]
    #create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')
    #fit model to training data
    ensemble.fit(X_train, y_train)
    #test our model on the test data
    ensemble_score = ensemble.score(X_test, y_test)
    # 0.48034547959124885


    print(f"Oversmapling - SMOTE: Accuracy:{sm_accuracy}, F1 Score: {sm_f1_score}")
    print(f"Undersampling - Cluster Centroids: Accuracy:{cc_accuracy}, F1 Score: {cc_f1_score}")
    print(f"Random Sampling - Oversampling Strategy: Accuracy:{rand_over_samp_acc}, F1 Score: {rand_over_samp_f1score}")
    print(f"Random Sampling - Undersampling Strategy: Accuracy:{rand_under_samp_acc}, F1 Score: {rand_under_samp_f1score}")
    print(f"Ensemble models(KNN, RFC): Score:{ensemble_score}")

    # Crime Type - offense codes are grouped to T1, T2, and T3
    # Oversmapling - SMOTE: Accuracy:0.2820029275896812, F1 Score: 0.22177194754181842
    # Undersampling - Cluster Centroids: Accuracy:0.34109461437560595, F1 Score: 0.17350831881494372
    # Random Sampling - Oversampling Strategy: Accuracy:0.3030815732942988, F1 Score: 0.25179158696314446
    # Random Sampling - Undersampling Strategy: Accuracy:0.3776704750679619, F1 Score: 0.3332267683237674
    # Ensemble models(KNN, RFC): Score:0.4476208581259624

    # Crime Type - offense codes are grouped to T1, T2
    # Oversmapling - SMOTE: Accuracy:0.6014466252184161, F1 Score: 0.48167857557071425
    # Undersampling - Cluster Centroids: Accuracy:0.6072168718761429, F1 Score: 0.4588208796746159
    # Random Sampling - Oversampling Strategy: Accuracy:0.6073658688555678, F1 Score: 0.4603966678278045
    # Random Sampling - Undersampling Strategy: Accuracy:0.5797201565822802, F1 Score: 0.5827553879113659
    # Ensemble models(KNN, RFC): Score:0.6336096550042667we `SW3`