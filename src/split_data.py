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
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

def main(spark, netID):
	path_small = f'hdfs:/user/{netID}/ml-latest-small/ratings.csv'
	path_large = '/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/ratings.csv'


	#Read
	ratings_small = spark.read.csv(path_small, schema='userId INT, movieId INT, rating FLOAT, timestamp TIMESTAMP')
	ratings_small.printSchema()
	
	# Give the dataframe a temporary view so we can run SQL queries
	ratings_small.createOrReplaceTempView('ratings_small')

	#Obtain userIds
	select_uids = spark.sql('SELECT distinct(userId) FROM ratings_small where userId is not null order by userId').collect()

	#Iterate 
	subdf_of_userId = spark.sql(f'SELECT * FROM ratings_small where userId={select_uids[0][0]}')
	train_ratings_small,test_ratings_small,val_ratings_small = subdf_of_userId.randomSplit([0.8, 0.1,0.1], seed=30)


	q = spark.sql('select count(*) from ratings_small').show()
	first = False
	for row in select_uids:
		if not first:
			first = True
			continue
		userId = row[0]
		subdf_of_userId = spark.sql(f'SELECT * FROM ratings_small where userId={userId}')
		train,test,val = subdf_of_userId.randomSplit([0.8, 0.1,0.1], seed=30)


		train_ratings_small = train_ratings_small.union(train)
		test_ratings_small = test_ratings_small.union(test)
		val_ratings_small = val_ratings_small.union(val)
	#80004 10311 10521
	print(train_ratings_small.count(),test_ratings_small.count(),val_ratings_small.count())
	train_ratings_small.write.parquet(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small.parquet')
	test_ratings_small.write.parquet(f'hdfs:/user/{netID}/ml-latest-small/test_ratings_small.parquet')
	val_ratings_small.write.parquet(f'hdfs:/user/{netID}/ml-latest-small/val_ratings_small.parquet')	


def randomSplit(userId):
	subdf_of_userId = spark.sql(f'SELECT * FROM ratings_small where userId={userId}')
	train,test,val = subdf_of_userId.randomSplit([0.8, 0.1,0.1], seed=30)
	return train,test,val


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
