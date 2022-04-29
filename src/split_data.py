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
	# path_large = '/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/ratings.csv'


	#Read
	ratings_small = spark.read.csv(path_small, header=True,schema='userId INT, movieId INT, rating FLOAT, timestamp BIGINT')
	ratings_small.printSchema()
	
	# Give the dataframe a temporary view so we can run SQL queries
	ratings_small.createOrReplaceTempView('ratings_small')

	#Obtain userIds
	select_uids = spark.sql('SELECT distinct(userId) FROM ratings_small where userId is not null order by userId').collect()



	#Iterate 
	train_ratings_small = None
	test_ratings_small = None
	val_ratings_small = None


	for row in select_uids:
		userId = row[0]
		# print(userId)
		subdf_of_userId = spark.sql(f'SELECT * FROM ratings_small where userId={userId}')
		subdf_of_userId = subdf_of_userId.select(col('userId'),col('movieId'),col('rating'),from_unixtime(col('timestamp'),'yyyy-MM-dd HH:mm:ss').alias('date'),from_unixtime(col('timestamp'),'yyyy').alias('Year').cast(IntegerType()))


		filtered_train = subdf_of_userId.filter(col('Year')!=2018)
		filtered_train = filtered_train.select(col('userId'),col('movieId'),col('rating'),col('date'))
		if not train_ratings_small:
			train_ratings_small = filtered_train
		else:
			train_ratings_small = train_ratings_small.union(filtered_train)
		if random.choice([0,1]) == 0: # Val
			filtered_val = subdf_of_userId.filter(col('Year')==2018)
			filtered_val = filtered_val.select(col('userId'),col('movieId'),col('rating'),col('date'))
			if not val_ratings_small:
				val_ratings_small = filtered_val
			else:
				val_ratings_small = val_ratings_small.union(filtered_val)
		else: # Test
			filtered_test = subdf_of_userId.filter(col('Year')==2018)
			filtered_test = filtered_test.select(col('userId'),col('movieId'),col('rating'),col('date'))
			if not test_ratings_small:
				test_ratings_small = filtered_test
			else:
				test_ratings_small = test_ratings_small.union(filtered_test)


	print(train_ratings_small.count(),test_ratings_small.count(),val_ratings_small.count())
	train_ratings_small.write.parquet(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small.parquet')
	test_ratings_small.write.parquet(f'hdfs:/user/{netID}/ml-latest-small/test_ratings_small.parquet')
	val_ratings_small.write.parquet(f'hdfs:/user/{netID}/ml-latest-small/val_ratings_small.parquet')	


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
