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

    #FOR THE SMALL DATASET:
    #reading the files
    ratings_small = spark.read.csv(f'hdfs:/user/{netID}/ml-latest-small/ratings.csv', header=True,schema='userId INT, movieId INT, rating FLOAT, timestamp BIGINT')
     
    train_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small.parquet')
    val_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/val_ratings_small.parquet')
    test_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/test_ratings_small.parquet')

    movies_small = spark.read.csv(f'hdfs:/user/{netID}/ml-latest-small/movies.csv',header=True, schema='movieId INT, title string, genres string')

    # For the most popular movies
	# we use the number of ratings as a proxy for the number of views
    most_popular_small = ratings_small.groupBy("movieId").agg(count("userId").alias('num_ratings'),avg("rating")).join(movies_small,'movieId').sort(col("num_ratings").desc(),col('avg(rating)').desc())

    #Join for predictions of Val_Rating
    preds_small = val_ratings_small.join(most_popular_small,'movieId')
    preds_small.show(5)

    #RMSE score
    evaluator_small = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="avg(rating)")

    rmse_small = evaluator_small.evaluate(preds_small)
    
    print("The RMSE score for the small dataset is: {}".format(rmse_small))


    #FOR THE LARGE DATASET:
    #reading the files
    ratings_large = spark.read.csv(f'hdfs:/user/{netID}/ml-latest/ratings.csv', header=True,schema='userId INT, movieId INT, rating FLOAT, timestamp BIGINT')
    
    train_ratings_large = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest/train_ratings_big.parquet')
    val_ratings_large = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest/val_ratings_big.parquet')
    test_ratings_large = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest/test_ratings_big.parquet')

    movies_large = spark.read.csv(f'hdfs:/user/{netID}/ml-latest/movies.csv', header=True, schema='movieId INT, title string, genres string')

    # For the most popular movies
	# we use the number of ratings as a proxy for the number of views
    most_popular_large = ratings_large.groupBy("movieId").agg(count("userId").alias('num_ratings'),avg("rating")).join(movies_large,'movieId').sort(col("num_ratings").desc(),col('avg(rating)').desc())

    #Join for predictions of Val_Rating
    preds_large = val_ratings_large.join(most_popular_large,'movieId')
    preds_large.show(5)

    #RMSE_score:
    evaluator_large = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="avg(rating)")
    
    rmse_large = evaluator_large.evaluate(preds_large)

    print("The RMSE score for the large dataset is: {}".format(rmse_large))


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)




