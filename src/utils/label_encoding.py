import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from itertools import chain


def stringToNumeric(df, features_list, columnName):
    if columnName in features_list:
        # Create a new column
        newColumnName = columnName+"_idx"
        df = df.withColumn(newColumnName, lit(0))

        # Build MAP with string to numeric values
        c_list = df.select(columnName).distinct().rdd.flatMap(lambda x: x).collect()
        c_map = {}
        for idx, val in enumerate(c_list):
            c_map[val] = idx
        
        # Run UDF
        mapping_expr = create_map([lit(x) for x in chain(*c_map.items())])
        df = df.withColumn(newColumnName, mapping_expr[df[columnName]])

        # Rename columns
        df = df.withColumnRenamed(columnName, columnName + "_org").\
                withColumnRenamed(newColumnName, columnName)

        df = df.drop(columnName + "_org")
    return df
    
def label_encoding(df, features_list):
    xy_columns = features_list + ["label"]
    df = stringToNumeric(df, xy_columns,  "DISTRICT")
    df = stringToNumeric(df, xy_columns, "STREET")
    df = stringToNumeric(df, xy_columns, "label")
    df = stringToNumeric(df, xy_columns, "DAY_OF_WEEK")
    df = stringToNumeric(df, xy_columns, "DAY_SLOT")
    df = stringToNumeric(df, xy_columns, "UCR_PART")
    df = stringToNumeric(df, xy_columns, "OFFENSE_CODE_GROUP")
    return df