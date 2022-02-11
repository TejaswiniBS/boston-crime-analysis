from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding

def dtc(data_dir):
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
    dtc = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 15) #, maxDepth = 25, impurity='entropy', maxBins=128)
    dtcModel = dtc.fit(train_data)
    predictions = dtcModel.transform(test_data)
    results(predictions)
    # print(df.columns)
    # print(df.select("label").distinct().show())



    #############
    # Experiments
    #############
    #  
    # The features are formatted as a single vector

    #*************************************************************************************************************************************
    # feature_list =  ['Lat','Long', 'Season'] # 0.18%
    # feature_list =  ['Lat','Long'] # 0.18.299%
    # feature_list = ['DISTRICT','STREET'] # 0.18.299%
    # feature_list = ['Season'] # 0.18.299%
    # DAY_SLOT in float = 16.19%
    # DAY_SLOT in int = 16.76%

    # feature_list = ['Lat','Long'] # 0.3914813799845119
    # feature_list = ['DISTRICT','REPORTING_AREA', 'STREET', 'YEAR','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night'] #0.43576704230276697
    # feature_list = ['DISTRICT', 'DAY_OF_WEEK', 'STREET', 'MONTH', 'DAY_SLOT'] #0.4045105525060897
    # feature_list = ['DISTRICT', 'DAY_OF_WEEK', 'STREET', 'MONTH', 'DAY_SLOT', 'Day','Night'] #0.4072536441297834
    # feature_list = ['DISTRICT', 'STREET'] # 0.3681085979438242
    # feature_list = ['YEAR', 'DAY_OF_WEEK', 'MONTH', 'DAY_SLOT', 'Day','Night', 'Season'] # 0.34521882174582724
    # feature_list = ['YEAR', 'DAY_OF_WEEK', 'MONTH', 'Lat','Long'] #  0.42685750115197013
    # feature_list = ['DAY_OF_WEEK', 'MONTH', 'Lat','Long'] #  Accuracy = 0.41226927183722883

    # feature_list = ['DISTRICT','REPORTING_AREA', 'STREET', 'YEAR','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night', 'UCR_PART'] #0.5676755277353196
    # feature_list = ['Lat','Long', 'UCR_PART'] #0.5768713161484667

    # UCR prediction
    # df = df.withColumnRenamed("label", "crime_code_group")
    # df = df.withColumnRenamed("UCR_PART", "label")
    # feature_list = ['DISTRICT','REPORTING_AREA', 'STREET', 'YEAR','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night', 'crime_code_group'] 
    # # Accuracy = 0.79 (with all severities)
    # # Accuracy = 0.8096547812418184 (with 1 &2 severity)
    #*************************************************************************************************************************************

    # df = cleanse_data()
    # df = encoding(df)

    # # OHE: Test1 # Accuracy = 0.42820520317476274
    # categorical_features = ['DISTRICT','REPORTING_AREA', 'STREET', 'YEAR','MONTH','DAY_OF_WEEK','HOUR']
    # numerical_features = ['Lat','Long']

    # # OHE: Test1 # Accuracy = Accuracy = 0.42751500310536505
    # categorical_features = ['DISTRICT','REPORTING_AREA','STREET', 'DAY_OF_WEEK']
    # numerical_features = [ 'YEAR','MONTH','HOUR','Lat','Long']

    # one_hot_encoders = one_hot_encoding(categorical_features) 
    # assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in one_hot_encoders] + numerical_features, outputCol="features")
    # pipeline = Pipeline(stages=one_hot_encoders+[assembler])
    # p=pipeline.fit(df)
    # df = p.transform(df)
    #*************************************************************************************************************************************

    # ------------------------------------------------------------------------------------------------
    # Predicting Crime Type
    ## ------------------------------------------------------------------------------------------------
    # space1_list = ['Lat','Long'] # Accuracy = 0.4213048277548046
    # space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.43070238702577374
    # time1_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.3754387746721368
    # time2_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR', 'Day', 'Night', 'Season', 'DAY_SLOT'] # Accuracy =  0.37315864300831664
    # space1_time_list = space1_list + time1_list # Accuracy = 0.41996862818546277
    # space2_time_list = space2_list + time1_list # Accuracy = 0.4261256837956273
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.43813544811573546(maxDepth = 25, impurity='gini', maxBins=32), Accuracy = 0.437863017176385(maxDepth = 25, impurity='entropy', maxBins=128)

    # # Only Type1 & Type2
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.586394911544595(maxDepth = 25, impurity='gini', maxBins=32), Accuracy = 0.6039994786570705(maxDepth = 25, impurity='entropy', maxBins=128)

    # # Only Type2 & Type3
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.5729563786123703

    # # Only Type1, deviced into 7 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.4859548676037495

    # # Only Type2, deviced into 22 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.27940127136915927

    # # Only Type 1 & 2, deviced into 29 categories
    # all = space1_list  + space2_list + time1_list # Accuracy = 0.228568359461674

    # features_list = all
    # prediction_type='OFFENSE_CODE_GROUP'
    #*************************************************************************************************************************************
