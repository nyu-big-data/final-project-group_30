import getpass
from pyspark.sql import SparkSession

import pandas as pd
import numpy as np

from pyspark.sql.functions import *
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType,StructField,DateType,FloatType,IntegerType

import seaborn as sns
sns.set_style('whitegrid')

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def main(spark, netID):
        
        #rating
        path_small_ratings = f'hdfs:/user/{netID}/ml-latest-small/ratings.csv'
        ratings_small = spark.read.csv(path_small_ratings, header=True,schema='userId INT, movieId INT, rating FLOAT, timestamp BIGINT')
        ratings_small.show(5)

        pd_ratings = ratings_small.toPandas()
        print(pd_ratings)

        plt.figure(figsize=(10,4))
        pd_ratings['rating'].hist(bins=10)


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('EDA_small').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
