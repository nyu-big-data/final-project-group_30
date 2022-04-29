
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
#Importing for groupby
# from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql import DataFrameNaFunctions as DFna
from pyspark.sql.functions import udf, col, when
import matplotlib.pyplot as plt
import pyspark as ps
import os, sys, requests, json
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import Row
import numpy as np
import math

def main(spark, netID):

	ratings = spark.read.csv(f'hdfs:/user/{netID}/ml-latest-small/ratings.csv', header=True,schema='userId INT, movieId INT, rating FLOAT, timestamp BIGINT')

	train_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small.parquet')
	val_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/val_ratings_small.parquet')
	test_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/test_ratings_small.parquet')


	movies = spark.read.csv(f'hdfs:/user/{netID}/ml-latest-small/movies.csv',header=True, schema='movieId INT, title string, genres string')

	# Most popular movies
	# we use the number of ratings as a proxy for the number of views
	most_popular = ratings.groupBy("movieId").agg(count("userId").alias('num_ratings'),avg("rating")).join(movies,'movieId').sort(col("num_ratings").desc(),col('avg(rating)').desc())

	#Join for predictions of Val_Rating
	preds = val_ratings_small.join(most_popular,'movieId')
	#RSME
	evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="avg(rating)")

	rmse = evaluator.evaluate(preds)
	print("the rmse score: {}".format(rmse))

if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)


