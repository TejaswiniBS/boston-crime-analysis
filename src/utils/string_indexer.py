import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer

def string_indexer(df):
    # Convert String Columns To Numeric Columns
    def stringToNumeric(df_t, column, column_indexer):
        # Apply StringIndexer to crime_code_group column
        col_indexer = StringIndexer(inputCol=column, outputCol=column_indexer)
        #Fits a model to the input dataset with optional parameters.
        return col_indexer.fit(df_t).transform(df_t)

    # PY13: DISTRICT: string column to numeric value (After this step = 214932)
    df = stringToNumeric(df, "DISTRICT", "DISTRICT_INDEXER")

    # PY14: DAY_OF_WEEK: string column to numeric value (After this step = 214932)
    df = stringToNumeric(df, "DAY_OF_WEEK", "DAY_OF_WEEK_INDEXER")

    # PY15: crime_code_group: string column to numeric value (After this step = 214932)
    df = stringToNumeric(df, "crime_code_group", "crime_code_group_indexer")

    # PY16: Select Required colums for modeling (After this step = 214932)
    df = df.select('crime_code_group_indexer','DISTRICT_INDEXER','REPORTING_AREA','MONTH','DAY_OF_WEEK_INDEXER', 'YEAR', 'HOUR','Lat','Long', 'Day','Night')

    # PY17: All years Distinct rows (After distinct = 200343)
    # 2020 - 26624
    df = df.distinct()

    # PY18: Rename columns
    df = df.withColumnRenamed("crime_code_group_indexer", "label").\
            withColumnRenamed("DISTRICT_INDEXER", "DISTRICT").\
            withColumnRenamed("DAY_OF_WEEK_INDEXER", "DAY_OF_WEEK")

    return df