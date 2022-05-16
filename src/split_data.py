'''
Takes the input data ratings and splits into train, validation and test data
Stater code taken from lab3
Data Copied to hfs, since working on spark
spark-submit --conf  spark.dynamicAllocation.enabled=true --conf spark.shuffle.service.enabled=false --conf spark.dynamicAllocation.shuffleTracking.enabled=true
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
#Importing for groupby
from pyspark.sql.functions import *
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType,StructField,DateType,FloatType,IntegerType
import random

def main(spark, netID):
	path_small = f'hdfs:/user/{netID}/ml-latest-small/ratings.csv'
	path_large = f'hdfs:/user/{netID}/ml-latest/ratings.csv'
	store_small = 'ml-latest-small'
	store_big = 'ml-latest'
	small = 'small'
	big = 'big'

    #FOR SMALL DATASET
	#Read dataset
	ratings = spark.read.csv(path_small, header=True,inferSchema=True)
	ratings.printSchema()
	
	# Give the dataframe a temporary view so we can run SQL queries
	ratings.createOrReplaceTempView('ratings')

	#Obtain userIds
	select_uids = ratings.select('userId').distinct().collect()

	#Iterate 
	train_ratings = None
	test_ratings = None
	val_ratings = None


	for row in select_uids:
		userId = row[0]
		# print(userId)
		subdf_of_userId = spark.sql(f'SELECT * FROM ratings where userId={userId}')
		subdf_of_userId = subdf_of_userId.select(col('userId'),col('movieId'),col('rating'),from_unixtime(col('timestamp'),'yyyy-MM-dd HH:mm:ss').alias('date'),from_unixtime(col('timestamp'),'yyyy').alias('Year').cast(IntegerType()))


		filtered_train = subdf_of_userId.filter(col('Year')!=2018)
		filtered_train = filtered_train.select(col('userId'),col('movieId'),col('rating'),col('date'))
		if not train_ratings:
			train_ratings = filtered_train
		else:
			train_ratings = train_ratings.union(filtered_train)
		if random.choice([0,1]) == 0: # Val
			filtered_val = subdf_of_userId.filter(col('Year')==2018)
			filtered_val = filtered_val.select(col('userId'),col('movieId'),col('rating'),col('date'))
			if not val_ratings:
				val_ratings = filtered_val
			else:
				val_ratings = val_ratings.union(filtered_val)
		else: # Test
			filtered_test = subdf_of_userId.filter(col('Year')==2018)
			filtered_test = filtered_test.select(col('userId'),col('movieId'),col('rating'),col('date'))
			if not test_ratings:
				test_ratings = filtered_test
			else:
				test_ratings = test_ratings.union(filtered_test)


	print(train_ratings.count(),test_ratings.count(),val_ratings.count())
	train_ratings.write.mode("overwrite").parquet(f'hdfs:/user/{netID}/{store_small}/train_ratings_{small}.parquet')
	test_ratings.write.mode("overwrite").parquet(f'hdfs:/user/{netID}/{store_small}/test_ratings_{small}.parquet')
	val_ratings.write.mode("overwrite").parquet(f'hdfs:/user/{netID}/{store_small}/val_ratings_{small}.parquet')	


    #FOR LARGE DATASET
    #Read
	ratings_large = spark.read.csv(path_large, header=True,inferSchema=True)
	ratings_large.printSchema()
	
	# Give the dataframe a temporary view so we can run SQL queries
	ratings_large.createOrReplaceTempView('ratings_large')

	#Obtain userIds
	query_large = spark.sql('SELECT distinct userId, movieId, rating FROM ratings_large')

	#Iterate 
	train_large_ratings = None
	test_large_ratings = None
	val_large_ratings = None

	weights = [.7, .2, .1]
	seed = 40

	train_large_ratings, val_large_ratings, test_large_ratings = query_large.randomSplit(weights, seed)

	print(train_large_ratings.count(), val_large_ratings.count(), test_large_ratings.count())
	train_large_ratings.write.mode("overwrite").parquet(f'hdfs:/user/{netID}/{store_big}/train_ratings_{big}.parquet')
	test_large_ratings.write.mode("overwrite").parquet(f'hdfs:/user/{netID}/{store_big}/test_ratings_{big}.parquet')
	val_large_ratings.write.mode("overwrite").parquet(f'hdfs:/user/{netID}/{store_big}/val_ratings_{big}.parquet')	


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
