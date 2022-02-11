from datetime import time
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from prettytable import PrettyTable


def checkNumeric(df):
    x = PrettyTable()
    x.field_names = ["Column", "Total Count", "Numeric Count", "Non Numeric Count", "IsNumeric"]
    for column in df.columns:
        df_t = df.select(col(column), col(column).cast("int").isNotNull().alias("Value "))
        total_rec = df_t.select(column).count()
        numeric_count = df_t.where(col('Value ') == True).count()
        non_numeric_count = df_t.where(col('Value ') == False).count()
        x.add_row([column, total_rec, numeric_count, non_numeric_count, total_rec == numeric_count])
    print(x)

# >>> df.select("OFFENSE_CODE").describe().show()
# +-------+-----------------+
# |summary|     OFFENSE_CODE|
# +-------+-----------------+
# |  count|           444138|
# |   mean|2297.703760993205|
# | stddev|1193.330196135051|
# |    min|              100|
# |    max|            99999|
# +-------+-----------------+
def print_crime_offense_range_counts(df): 
    x = PrettyTable()
    x.field_names = ["Range", "Count"]
    # Last valid number is 3831, and 99999 is irrelevant
    for i in range(100, 3900, 100):
        r_e = i+100
        count = df.select("OFFENSE_CODE").where((col("OFFENSE_CODE") >= i) & (col("OFFENSE_CODE") < r_e)).count()
        if count> 0:
            x.add_row([f'{i}-{r_e}', count])
    print(x)

# Lat & Long
# Remove records based on Location (boston outliers)
# df.select("Lat", "Long").describe().show()
# +-------+------------------+-------------------+
# |summary|               Lat|               Long|
# +-------+------------------+-------------------+
# |  count|            517495|             517495|
# |   mean|36.491685593995115| -61.29479660043357|
# | stddev|14.588936923566086| 24.492251318003127|
# |    min|              -1.0|       -71.17867378|
# |    max|  42.3950415802391|5.24969124614893E-8|
# +-------+------------------+-------------------+

#  df.select("Lat").where(col("Lat") < 42).distinct().show(n=10, truncate=False)
# +-------------------+
# |Lat                |
# +-------------------+
# |0.0                |
# |-1.0               |
# |1.32727612912185E-7|
# +-------------------+

# df.select("Long").where(col("Long") > -70).distinct().show(n=10, truncate=False)
# +-------------------+
# |Long               |
# +-------------------+
# |0.0                |
# |-1.0               |
# |5.24969124614893E-8|
# +-------------------+

# Drop outliers based on min and max values
# df = df.filter( (df.Lat  > 42) & (df.Long  < -70) )
# df.select("Lat", "Long").describe().show()
# +-------+-------------------+-------------------+
# |summary|                Lat|               Long|
# +-------+-------------------+-------------------+
# |  count|             446222|             446222|
# |   mean| 42.322211447361106| -71.08325848286599|
# | stddev|0.03200683134754396|0.03023934957911105|
# |    min|   42.2077599789506|       -71.17867378|
# |    max|   42.3950415802391|  -70.9537263645748|
# +-------+-------------------+-------------------+
def cleanse_lat_long(df, features_list):
    if "Lat" in features_list or "Long" in features_list:
        # Lat & Long drop missing values (After this step = 517495)
        df = df.na.drop(subset=["Lat"])

        # Drop outliers based on min and max values (After this step = 446222)
        df = df.filter( (df.Lat  > 42) & (df.Long  < -70) )
    return df

def cleanse_crime_code_group_manualy_categorise(df, prediction_type):
    if prediction_type == 'OFFENSE_CODE_GROUP':
        df = df.withColumn("OFFENSE_CODE", col("OFFENSE_CODE").cast("int"))
        # Filter records to consider only crime entries (After this step = 215355)
        # Interested only for 3625, since this format contains a list of start and end, hence end is start + 1
        CRIME_OFFENSE_CODES = {
            "Murder" : [[100, 200]],
            "TheftFraud": [[300, 400], [500, 800], [1000, 1400]],
            "Assault":[[400, 500], [800, 900],[1600, 1700], [2000, 2100], [3625, 3626]], 
            "Vandalism": [[900, 1000], [1400, 1500]],
            "Kidnapping": [[2500, 2600]],
            "Violation": [[1500, 1600], [1800, 1900], [2100, 2300], [2600, 2700], [2900, 3000]]
        }
        def update_crime_group(offense_code):
            if offense_code <= 3625:
                for crime_type, crime_codes in CRIME_OFFENSE_CODES.items():
                    for code_range in crime_codes:
                        if offense_code >= code_range[0] and offense_code < code_range[1]:
                            return crime_type
            return None
        update_crime_group_udf = udf(lambda x: update_crime_group(x), StringType())
        df = df.withColumn('label', update_crime_group_udf(col('OFFENSE_CODE')))
        # Select only valid entries
        df = df.where(col("label").isNotNull())
    return df

# >>> df.where(col('UCR_PART') == 'Part One').select('OFFENSE_CODE').describe().show()
# +-------+-----------------+
# |summary|     OFFENSE_CODE|
# +-------+-----------------+
# |  count|            67626|
# |   mean|564.4874012953597|
# | stddev|107.8126237736896|
# |    min|              111|
# |    max|              770|
# +-------+-----------------+

# >>> df.where(col('UCR_PART') == 'Part Two').select('OFFENSE_CODE').describe().show()
# +-------+------------------+
# |summary|      OFFENSE_CODE|
# +-------+------------------+
# |  count|            107173|
# |   mean|1791.7216836330047|
# | stddev| 723.4502968367897|
# |    min|               801|
# |    max|              2914|
# +-------+------------------+

# >>> df.where(col('UCR_PART') == 'Part Three').select('OFFENSE_CODE').describe().show()
# +-------+------------------+
# |summary|      OFFENSE_CODE|
# +-------+------------------+
# |  count|            175361|
# |   mean|3332.3024788864113|
# | stddev| 298.4273316150197|
# |    min|              3001|
# |    max|              3831|
# +-------+------------------+

# >>> df.where(col('UCR_PART') == 'Other').select('OFFENSE_CODE').describe().show()
# +-------+-----------------+
# |summary|     OFFENSE_CODE|
# +-------+-----------------+
# |  count|             1387|
# |   mean|782.0223503965393|
# | stddev| 292.239067648856|
# |    min|              112|
# |    max|             2631|
# +-------+-----------------+
def cleanse_crime_code_group_using_severity(df, prediction_type):
    if prediction_type == 'OFFENSE_CODE_GROUP':
        df = df.withColumn("OFFENSE_CODE", col("OFFENSE_CODE").cast("int"))

        # First, remove Severity corresponding to 'Other'
        df = df.filter(df['UCR_PART'] !='Other')

        # Second, categorise based on severity
        # Interested only for 3625, since this format contains a list of start and end, hence end is start + 1
        # "Type1" : [[100, 800]],
        #  "Type3":[[3000, 3626]]
        CRIME_OFFENSE_CODES = {
            "Severe": [[100, 800]],
            "Major": [[800, 3000]],
            "Minor": [[3000, 3626]]
        }

        def update_crime_group(offense_code):
            if offense_code <= 3625:
                for crime_type, crime_codes in CRIME_OFFENSE_CODES.items():
                    for code_range in crime_codes:
                        if offense_code >= code_range[0] and offense_code < code_range[1]:
                            return crime_type
            return None
        update_crime_group_udf = udf(lambda x: update_crime_group(x), StringType())
        df = df.withColumn('label', update_crime_group_udf(col('OFFENSE_CODE')))
        # Select only valid entries
        df = df.where(col("label").isNotNull())
    return df

def cleanse_reporting_area(df, features_list):
    if "REPORTING_AREA" in features_list:
        df = df.where(col("REPORTING_AREA") != ' ')
        df = df.withColumn("REPORTING_AREA", col("REPORTING_AREA").cast("int"))
    return df

def cleanse_ucr_part(df, prediction_type):
    if prediction_type == 'UCR_PART':
        df = df.withColumn("OFFENSE_CODE", col("OFFENSE_CODE").cast("int"))

        # First, remove Severity corresponding to 'Other'
        df = df.filter(df['UCR_PART'] !='Other')
        # df = df.filter(df['UCR_PART'] !='Part Three')
        # df = df.filter(df['UCR_PART'] !='Part One')

        # Remove all records with null value (After this step = 214932)
        df = df.na.drop(subset=['UCR_PART'])
       
        # Rename column
        df = df.withColumnRenamed("UCR_PART", "label")
    return df

def add_day_night_columns(df, features_list):
    if 'MONTH' in features_list and \
        'HOUR' in features_list:
        TIME_DAY_RANGE = {
            1:[ 6, 18], 2:[ 6, 19], 3:[ 6, 20], 4:[ 5, 20],
            5:[ 5, 21], 6:[ 4, 21], 7:[ 5, 21], 8:[ 5, 21],
            9:[ 6, 20], 10:[ 6, 19], 11:[ 6, 17], 12:[ 6, 17]
        }
        def is_day(month, hour):
            hours_range = TIME_DAY_RANGE[month]
            if hour >= hours_range[0] and hour <= hours_range[1]:
                return 1
            return 0
        update_day_udf = udf(lambda x,y: is_day(x,y), IntegerType())
        df = df.withColumn('Day', update_day_udf(col('MONTH'), col('HOUR')))
        df = df.withColumn('Night', when((col('Day') == 0 ),1).otherwise(0))
    return df

def add_seasons(df, features_list):
    if 'MONTH' in features_list:
        def dateToSeason(month):
            if month >= 3 and month <= 5:
                return 0
            if month >=6 and month <= 8:
                return 1
            if month >= 9 and month <= 10:
                return 2
            return 3
        update_season = udf(lambda x: dateToSeason(x), IntegerType())
        df = df.withColumn('Season', update_season(col('MONTH')))
    return df

def add_day_slot(df, features_list):
    if 'HOUR' in features_list:
        def hourToSlot(hour):
            return int((hour/4)+1)

        update_slot = udf(lambda x: hourToSlot(x), IntegerType())
        df = df.withColumn('DAY_SLOT', update_slot(col('HOUR')))
    return df

def drop_null_values(df, features_list):
    xy_columns = features_list + ["label"]
    return df.na.drop(subset=xy_columns)

def drop_unused_columns(df):
    df = df.drop('INCIDENT_NUMBER', 'OFFENSE_DESCRIPTION',
                 'SHOOTING', 'OCCURRED_ON_DATE', 'Location')
    return df

def drop_unwanted_columns(df, features_list):
    remove_columns = []
    xy_columns = features_list + ['label']
    for c in df.columns:
        if c not in xy_columns:
            remove_columns.append(c)
        
    if remove_columns:
        df = df.drop(*remove_columns)
    return df

def print_null_values(df):
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show(vertical=True)

def cleanse_data(data_dir, features_list, prediction_type):
    spark = SparkSession.builder.appName('BostonCrimeData_2015_to_2021').getOrCreate()

    # Load CSV file (Initial records = 585503, 17)
    df = spark.read.option("header", "true").option("nullValue", "?").option("inferSchema", "true").csv(f"{data_dir}/*.csv")
    df = df.distinct()
    df = drop_unused_columns(df)

    # Cleanse features based on the dimension of data required for model
    # If it's time, just drop null values
    #   YEAR           | 6038
    #   MONTH          | 6038
    #   DAY_OF_WEEK    | 6038
    #   HOUR           | 6038
    # If it's space, 
    #   Street Address
    #       DISTRICT       | 9432
    #       REPORTING_AREA | 6038
    #       STREET         | 17231
    #   Co-ordinates - Drop outliers and null values
    #       Lat            | 43427
    #       Long           | 43427

    # Prediction Types
    # Crime Type (CRIME_CODE_GROUP)
    #   Create groups using crime code value and offense_codes.csv
    #   Create groups using severity range using UCR_PART - Consider only Part One, Two, and Three
    # Crime Severity(UCR_PART)
    #   Drop null values

    # Cleanse columns
    # Cleanse Co-ordinates
    df = cleanse_lat_long(df=df, features_list=features_list)
    
    df = cleanse_crime_code_group_manualy_categorise(df=df, prediction_type=prediction_type)
    # df = cleanse_crime_code_group_using_severity(df=df, prediction_type=prediction_type)

    df = cleanse_ucr_part(df=df, prediction_type=prediction_type)

    # Cleanse REPORTING_AREA part of STREET address
    df = cleanse_reporting_area(df=df, features_list=features_list)

    # New columns
    df = add_day_night_columns(df=df, features_list=features_list)
    df = add_seasons(df=df, features_list=features_list)
    df = add_day_slot(df=df, features_list=features_list)

    # Final cleanup
    df = drop_unwanted_columns(df=df, features_list=features_list)
    df = drop_null_values(df=df, features_list=features_list)

    # print_null_values(df)
    df.columns

    return df


if __name__ == '__main__':
    space1_list = ['Lat','Long'] # After cleaning, total valid records: 293760
    space2_list = ['DISTRICT','REPORTING_AREA','STREET'] # After cleaning, total valid records: 294262
    time_list = ['YEAR', 'MONTH','DAY_OF_WEEK','HOUR'] # After cleaning, total valid records: 310091
    space1_time_list = space1_list + time_list # After cleaning, total valid records: 293760
    space2_time_list = space2_list + time_list # After cleaning, total valid records: 294262
    all = space1_list + space2_list + time_list # After cleaning, total valid records: 292238
    # ------------------------------------------------------------------------------------------------
    # features_list = all 
    # prediction_type = "OFFENSE_CODE_GROUP"

    features_list = all + ['OFFENSE_CODE_GROUP']
    prediction_type = 'UCR_PART'

    df = cleanse_data(features_list=features_list, prediction_type=prediction_type)
