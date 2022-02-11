from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding
from utils.regression_evaluator import regres_evaluator

def lr(data_dir):
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

    # The features are formatted as a single vector
    # feature_list = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']
    # feature_list = ['Day', 'Night','Lat','Long']
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")

    # RUNNING THE MODEL
    # 10% of the data is held out for testing with the remaining 90% used for training. 
    # Random sampling should be sufficient for this particular dataset.
    (trainingData, testData) = df.randomSplit([0.9, 0.1])

    train01 = assembler.transform(trainingData)
    train02 = train01.select("features","label")
    train02.show(n=5, truncate=False)

    lr = LinearRegression()
    model = lr.fit(train02)

    test01 = assembler.transform(testData)
    test02 = test01.select('features', 'label')
    predictions = model.transform(test02)
    predictions.show(n=5, truncate=False)

    # import chart_studio.plotly as py
    # import plotly.graph_objects as go
    # fig = go.Figure()
    # fig.add_trace( go.Scatter(x=x, y=y, mode='markers', name='Original_Test',))
    # fig.add_trace(go.Scatter(x=x, y=y_pred, name='Predicted'))
    # fig.update_layout(
    #     title="Linear Regression",
    #     xaxis_title="Features",
    #     yaxis_title="OffenseCodeGroup",
    #     font=dict(family="Courier New, monospace", size=18, color="#7f7f7f")
    # )
    # fig.show()

    regres_evaluator(predictions)

    # ['Day', 'Night','Lat','Long']
    # ra2: 0.005961694589501687
    # mse: 5.442902377764806
    # rmse: 2.3330028670717073
    # mae: 1.868562638218242


    # feature_list = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']
    # ra2: 0.01141375277025558
    # mse: 5.388498866217879
    # rmse: 2.3213140386896987
    # mae: 1.8538156527991962
