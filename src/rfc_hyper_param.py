from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding


def rfc_hyper_param(data_dir):

    # The features are formatted as a single vector
    # feature_list = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long'] # Accuracy = 0.4307378723328742

    space1_list = ['Lat','Long'] # Accuracy = 0.4307378723328742
    space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # Accuracy = 0.4307378723328742
    time_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # Accuracy = 0.4307378723328742
    space1_time_list = space1_list + time_list # Accuracy = 0.4307378723328742
    space2_time_list = space2_list + time_list # Accuracy = 0.4307378723328742
    all = space1_list  + space2_list + time_list

    # ------------------------------------------------------------------------------------------------
    features_list = all
    df = cleanse_data(data_dir=data_dir, features_list=features_list, prediction_type='OFFENSE_CODE_GROUP')
    df = encoding(df, features_list)
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")

    # RUNNING THE MODEL
    # 10% of the data is held out for testing with the remaining 90% used for training. 
    # Random sampling should be sufficient for this particular dataset.
    train_data, test_data = df.randomSplit([0.9, 0.1])

    rfc = RandomForestClassifier(labelCol="label", featuresCol="features")

    # Create two-stage workflow into an ML pipeline.
    pipeline = Pipeline(stages=[assembler, rfc])

    # HYPERPARAMETER GRID
    paramGrid = ParamGridBuilder() \
        .addGrid(rfc.numTrees, [20]) \
        .addGrid(rfc.maxBins, [5]) \
        .addGrid(rfc.maxDepth, [5]) \
        .addGrid(rfc.minInstancesPerNode, [5]) \
        .addGrid(rfc.impurity, ['entropy']) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=MulticlassClassificationEvaluator(),
                            numFolds=3)

    # The model is fit using the CrossValidator we created. 
    # This triggers Spark to assess the features and “grow” numerous decision trees using 
    # random samples of the training data. The results are recorded for each permutation of 
    # the hyperparameters.
    cvModel = crossval.fit(train_data)

    # The transformer (i.e. prediction generator) from out cross-validator by default applies 
    # the best performing pipeline. We can test our new model by making predictions on 
    # the hold out data.
    predictions = cvModel.transform(test_data)

    # predictions.select('label', 'prediction').show(25)

    # // Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %s" % (accuracy))

    # Accuracy = 0.2894860062464112
    # Test Error = 0.7105139937535888

    # Accuracy = 0.1962859112291088
    # Test Error = 0.8037140887708912


