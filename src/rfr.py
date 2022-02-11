import numpy as np
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt


from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding
from utils.regression_evaluator import regres_evaluator

def rfr(data_dir):
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


    # Select only 2020 records (After this step = 27908)
    # df = df.where(col("YEAR") == 2020)

    # The features are formatted as a single vector
    # feature_list = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']
    # feature_list = ['MONTH','DAY_OF_WEEK','HOUR', 'Lat','Long']
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")

    # RUNNING THE MODEL
    # 10% of the data is held out for testing with the remaining 90% used for training. 
    # Random sampling should be sufficient for this particular dataset.
    (trainingData, testData) = df.randomSplit([0.9, 0.1])

    # Create Random Forest Regressor model
    rf = RandomForestRegressor(featuresCol='features', labelCol='label')

    # Create two-stage workflow into an ML pipeline.
    pipeline = Pipeline(stages=[assembler, rf])

    # HYPERPARAMETER GRID
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 15, num = 3)]) \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 10, num = 3)]) \
        .build()

    # To evaluate our model and the corresponding “grid” of parameter variables, 
    # we use three folds cross-validation. This method randomly partitions the 
    # original sample into three subsamples and uses them for training and validation.
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=RegressionEvaluator(),
                            numFolds=3)

    # The model is fit using the CrossValidator we created. 
    # This triggers Spark to assess the features and “grow” numerous decision trees using 
    # random samples of the training data. The results are recorded for each permutation of 
    # the hyperparameters.
    cvModel = crossval.fit(trainingData)

    # The transformer (i.e. prediction generator) from out cross-validator by default applies 
    # the best performing pipeline. We can test our new model by making predictions on 
    # the hold out data.
    predictions = cvModel.transform(testData)

    predictions.select('label', 'prediction').show(25)

    # BEST HYPERPARAMETERS
    bestPipeline = cvModel.bestModel
    bestModel = bestPipeline.stages[1]
    print('numTrees - ', bestModel.getNumTrees)
    print('maxDepth - ', bestModel.getOrDefault('maxDepth'))
    regres_evaluator(predictions)

    # rmse = evaluator.evaluate(predictions)
    # rfPred = cvModel.transform(df)
    # rfResult = rfPred.toPandas()
    # plt.plot(rfResult.label, rfResult.prediction, 'bo')
    # plt.xlabel('Offense_Code_Group')
    # plt.ylabel('Prediction')
    # plt.suptitle("Model Performance RMSE: %f" % rmse)
    # plt.show()

    # FEATURE IMPORTANCE
    importances = bestModel.featureImportances
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, features_list, rotation=40)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.title('Feature Importances')
    plt.show()