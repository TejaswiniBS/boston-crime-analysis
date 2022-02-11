from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

var df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("./data/*.csv")

df.write.csv("boston_2015_to_2021.csv")
