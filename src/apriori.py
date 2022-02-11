from pyspark.sql.functions import *
from pyspark.sql.types import *

from utils.cleanse import cleanse_data
from utils.encoding import encoding
from utils.results import results
from utils.one_hot_encoding import one_hot_encoding

def apriori(data_dir):
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

    df = cleanse_data(data_dir=data_dir, features_list=features_list, prediction_type=prediction_type)

    # The features are formatted as a single vector
    # feature_list = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']

    MONTH_NUM_TO_STRING = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr",
        5: "may", 6: "jun", 7: "jul", 8: "aug",
        9: "sep", 10: "oct", 11: "nov", 12: "dec"
    }
    def monthToString(month):
        return MONTH_NUM_TO_STRING[month]
    update_month_udf = udf(lambda x: monthToString(x), StringType())
    df = df.withColumn('MONTH_NEW', update_month_udf(col('MONTH')))

    def hourToSlot(hour):
        return f"t{(int)(hour/4)+1}"
    update_slot = udf(lambda x: hourToSlot(x), StringType())
    df = df.withColumn('DAY_SLOT', update_slot(col('HOUR')))
    # df.show()

    df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("data_apriori.csv")
