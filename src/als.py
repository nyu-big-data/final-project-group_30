
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
from pyspark.ml.recommendation import ALS, ALSModel
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

	#Check if they are disjoint
	print('check disjoint')
	val_ratings_small.join(test_ratings_small,'userId').show()


	movies = spark.read.csv(f'hdfs:/user/{netID}/ml-latest-small/movies.csv',header=True, schema='movieId INT, title string, genres string')

	LOAD = True

	if not LOAD:
		seed = 5
		iterations = 10
		regularization_parameter = 0.1
		rank = 4

		als = ALS(maxIter=iterations, regParam=regularization_parameter, rank=rank, userCol="userId", itemCol="movieId", ratingCol="rating",seed=seed, nonnegative = True,coldStartStrategy="drop")
		paramGrid = ParamGridBuilder() \
	    .addGrid(als.regParam, [0.1, 0.01]) \
	    .addGrid(als.rank, [4,5,6,7,8,9,10,11,12]) \
	    .build()
		evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
		crossval = CrossValidator(estimator=als, \
	        estimatorParamMaps=paramGrid, \
	        evaluator=evaluator, \
	        numFolds=5)
		cvModel = crossval.fit(train_ratings_small)

		#Extract best model from the cv model above
		best_model = cvModel.bestModel


		cvModel_pred = best_model.transform(val_ratings_small) # Predict
		cvModel_pred.show(n=10) # Show 10 predictions
		cvModel_pred.join(movies,'movieId').select('userId','title','genres','prediction').show(5) # Show along with movie name
		cvModel_pred = cvModel_pred.filter(col('prediction') != np.nan) # New Predictions
		rmse = evaluator.evaluate(cvModel_pred)

		print("the rmse for optimal grid parameters with cross validation is: {}".format(rmse))

		print("**Best Model**")
		# Print "Rank"
		print("  Rank:", best_model._java_obj.parent().getRank())
		# Print "MaxIter"
		print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
		# Print "RegParam"
		print("  RegParam:", best_model._java_obj.parent().getRegParam())


		#Save
		best_model.write().overwrite().save(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small/als_model')
	else:
		best_model = ALSModel.load(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small/als_model')

	#100th User’s ALS Recommendations
	nrecommendations = best_model.recommendForAllUsers(5)
	nrecommendations = nrecommendations.withColumn("rec_exp", explode("recommendations")).select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
	nrecommendations.join(movies, on='movieId').filter('userId = 100').show(truncate=False)

	#100th User’s Actual Preference
	ratings.join(movies, on='movieId').filter('userId = 100').sort('rating', ascending=False).limit(10).show(truncate=False)

if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
