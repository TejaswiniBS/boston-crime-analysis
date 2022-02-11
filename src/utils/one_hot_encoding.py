import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder


def one_hot_encoding(columns):
    return [OneHotEncoder(dropLast=False, inputCol=c, outputCol= c+ "_one_hot") for c in columns]

