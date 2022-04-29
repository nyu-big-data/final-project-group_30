import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

MAXITER = 10

def basic_rec_train(spark):
	
	# Read training data	
	train = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small.parquet')
	
	# Start training
	# we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
	als = ALS(rank=2, maxIter=MAXITER, regParam = 3, seed=1, coldStartStrategy='drop', userCol='userId', itemCol='movieId', ratingCol='rating')
	model = als.fit(train)

	# Save the model
	model.write().overwrite().save(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small/{rank}_{regParam}_model')


if __name__ == '__main__':
	 
	spark = SparkSession.builder.appName('basic_rec_train').getOrCreate()
	
	#Get user netID from the command line
       

	basic_rec_train(spark)
