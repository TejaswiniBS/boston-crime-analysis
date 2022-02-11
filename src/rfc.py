from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding


def rfc(data_dir):
    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    space1_list = ['Lat','Long'] 
    space2_list = ['DISTRICT','REPORTING_AREA','STREET']
    time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR']
    all = space1_list  + space2_list + time1_list # Accuracy = 0.44446894663802444, F1 score: 0.42777359422852407


    features_list = all
    prediction_type='OFFENSE_CODE_GROUP'
    #*************************************************************************************************************************************
    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Severity
    ## ------------------------------------------------------------------------------------------------
    space1_list = ['Lat','Long'] # Accuracy =0.9352918880085079
    space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.9339876201291426
    time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.9490226698405932
    time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  0.8824156950714614
    space1_time_list = space1_list + time1_list # Accuracy = 0.9445001195976241
    space2_time_list = space2_list + time1_list # Accuracy = 0.927520330649868
    all = space1_list  + space2_list + time1_list # Accuracy = 0.9422184924918805

    # # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.9722794121823279

    # # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.97866593434881

    features_list = all + ['OFFENSE_CODE_GROUP']
    prediction_type='UCR_PART'
    #*************************************************************************************************************************************

    # Cleanse Data
    df = cleanse_data(data_dir=data_dir, features_list=features_list, prediction_type=prediction_type)
    df = encoding(df, features_list)

    assembler = VectorAssembler(inputCols=features_list, outputCol="features")
    df = assembler.transform(df)

    # Train:90%, Test:10%
    train_data, test_data = df.randomSplit([0.9, 0.1])

    rfc = RandomForestClassifier(labelCol="label", featuresCol="features", \
                                numTrees = 10, maxDepth = 5, maxBins = 5, impurity="entropy")
    # Train model with Training Data
    rfcModel = rfc.fit(train_data)
    predictions = rfcModel.transform(test_data)
    predictions.select('label', 'prediction', 'probability').show(10)
    results(predictions)
    # print(df.columns)
    # print(df.select("label").distinct().show())


    ### Experimentation

    # The features are formatted as a single vector
    # feature_list = ['DISTRICT','REPORTING_AREA', 'YEAR','MONTH','DAY_OF_WEEK','HOUR','STREET', 'Season', 'Lat','Long','Day','Night'] # 21.29
    # feature_list = ['YEAR','MONTH','DAY_OF_WEEK','HOUR', 'Season','Day','Night'] # 21.844
    # feature_list =  ['DISTRICT']  # 28%
    # feature_list =  ['Lat','Long'] # 39.26% (100 TREES)
    # feature_list =  ['Lat','Long'] # 0.390683463730224 (200 Trees)
    # feature_list = ['DISTRICT','REPORTING_AREA', 'STREET', 'YEAR','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night'] #0.4458993794314348, 0.43815451616915535

    
    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    # space1_list = ['Lat','Long'] # Accuracy = 0.41186044765209534
    # space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.41852292719704276
    # time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.3660693550337053
    # time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  0.35701362418813964
    # space1_time_list = space1_list + time1_list # Accuracy = 0.44197593019157577
    # space2_time_list = space2_list + time1_list # Accuracy = 0.4386492818437791
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.45198543039468436

    # # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.6208885096361371

    # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy =  0.5761006495576082

    # # # Only Type1, deviced into 7 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.5116929536732631

    # # # Only Type2, deviced into 22 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.29915618349300793

    # # # Only Type 1 & 2, deviced into 29 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.24221507402132147

    # features_list = all
    # prediction_type='OFFENSE_CODE_GROUP'
    #*************************************************************************************************************************************


