from pyspark.sql import SparkSession
import pyspark.sql.functions as func


var df = spark.read.format("csv").\
         option("header", "true").\
         option("inferSchema", "true").\
         load("./data/*.csv")

df.write.csv("boston_2015_to_2021.csv")

val parsed = spark.read.
 option("header", "true").
 option("nullValue", "?").
 option("inferSchema", "true").
 csv("./data/*.csv")
