from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler


from utils.regression_evaluator import regres_evaluator
from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results

def dtr(data_dir):
    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    space1_list = ['Lat','Long'] # Accuracy = 0.940065718398284
    space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.9514362917430098
    time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.9462569825822906
    time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  0.9338899416558533
    space1_time_list = space1_list + time1_list # Accuracy = 0.9402447881641893
    space2_time_list = space2_list + time1_list # Accuracy = 0.9440917838047351
    all = space1_list  + space2_list + time1_list # Accuracy = 0.9423429449312262

    # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.9796405840140008

    # # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.9641268757555908

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
    dtr = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'label', maxDepth = 15) #, maxDepth = 25, impurity='entropy', maxBins=128)
    dtrModel = dtr.fit(train_data)
    predictions = dtrModel.transform(test_data)
    predictions.show(n=5, truncate=False)

    regres_evaluator(predictions)
