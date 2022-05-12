import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
#Importing for groupby
from pyspark.sql.functions import *
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType,StructField,DateType,FloatType,IntegerType
import random

def main(spark, netID):
	path_large = f'hdfs:/user/{netID}/ml-latest/ratings.csv'
	

	#Read
	ratings_large = spark.read.csv(path_large, header=True,schema='userId INT, movieId INT, rating FLOAT, timestamp BIGINT')
	ratings_large.printSchema()
	
	# Give the dataframe a temporary view so we can run SQL queries
	ratings_large.createOrReplaceTempView('ratings_large')

	#Obtain userIds
	select_uids = spark.sql('SELECT distinct(userId) FROM ratings_large where userId is not null order by userId').collect()



	#Iterate 
	train_ratings_large = None
	test_ratings_large = None
	val_ratings_large = None


	for row in select_uids:
		userId = row[0]
		# print(userId)
		subdf_of_userId = spark.sql(f'SELECT * FROM ratings_large where userId={userId}')
		subdf_of_userId = subdf_of_userId.select(col('userId'),col('movieId'),col('rating'),from_unixtime(col('timestamp'),'yyyy-MM-dd HH:mm:ss').alias('date'),from_unixtime(col('timestamp'),'yyyy').alias('Year').cast(IntegerType()))


		filtered_train = subdf_of_userId.filter(col('Year')!=2018)
		filtered_train = filtered_train.select(col('userId'),col('movieId'),col('rating'),col('date'))
		if not train_ratings_large:
			train_ratings_large = filtered_train
		else:
			train_ratings_large = train_ratings_large.union(filtered_train)
		if random.choice([0,1]) == 0: # Val
			filtered_val = subdf_of_userId.filter(col('Year')==2018)
			filtered_val = filtered_val.select(col('userId'),col('movieId'),col('rating'),col('date'))
			if not val_ratings_large:
				val_ratings_large = filtered_val
			else:
				val_ratings_large = val_ratings_large.union(filtered_val)
		else: # Test
			filtered_test = subdf_of_userId.filter(col('Year')==2018)
			filtered_test = filtered_test.select(col('userId'),col('movieId'),col('rating'),col('date'))
			if not test_ratings_large:
				test_ratings_large = filtered_test
			else:
				test_ratings_large = test_ratings_large.union(filtered_test)


	print(train_ratings_large.count(),test_ratings_large.count(),val_ratings_large.count())
	train_ratings_large.write.parquet(f'hdfs:/user/{netID}/ml-latest/train_ratings_large.parquet')
	test_ratings_large.write.parquet(f'hdfs:/user/{netID}/ml-latest/test_ratings_large.parquet')
	val_ratings_large.write.parquet(f'hdfs:/user/{netID}/ml-latest/val_ratings_large.parquet')	


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
