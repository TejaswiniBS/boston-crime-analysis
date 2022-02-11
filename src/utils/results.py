import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



def printMetrics(predictions):
    preds_and_labels = predictions.select(['prediction','label']).withColumn('label_new', col('label').cast(FloatType())).orderBy('prediction')
    preds_and_labels = preds_and_labels.select(['prediction','label_new'])
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    conf_matrix = metrics.confusionMatrix().toArray()
    # Overall statistics
    # precision = metrics.precision(preds_and_labels.select(['label_new']))
    # recall = metrics.recall(preds_and_labels.select(['label_new']))
    # f1Score = metrics.fMeasure(preds_and_labels.select(['label_new']))
    # print("Summary Stats")
    # print("Precision = %s" % precision)
    # print("Recall = %s" % recall)
    # print("F1 Score = %s" % f1Score)
    return conf_matrix

def calAccuracy(predictions):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    return evaluator.evaluate(predictions)
    
def sampleOutput(predictions, verbose):
    if verbose:
        predictions.select('label', 'prediction', 'probability').show(10)

def f1Score(model, X_train, X_test, y_train, y_test):
    pred_train = model.predict(X_train)
    pred_i = model.predict(X_test)
    print(f'Train accuracy: {metrics.accuracy_score(y_train, pred_train)} \n'
        f'Test Accuracy: {metrics.accuracy_score(y_test, pred_i)}\n'
        f'F1 score: {metrics.f1_score(y_test, pred_i, average = "weighted")}')

def results(predictions, verbose=False):
    sampleOutput(predictions, verbose)
    conf_matrix = printMetrics(predictions)
    accuracy = calAccuracy(predictions)

    from sklearn.metrics import confusion_matrix
    import pandas as pd
    y_true = predictions.select('label').toPandas()
    y_pred = predictions.select('prediction').toPandas()
    unique_label = np.unique([y_true, y_pred])
    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print("\n**************************************************")
    print("Confusion Matrix:")
    print(cmtx)

    print("Accuracy = %s" % (accuracy))
    # print(conf_matrix)
    print("**************************************************")
