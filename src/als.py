#Use getpass to obtain user netID
import getpass

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import  RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from time import time 
from pyspark import SparkContext
from pyspark.sql.functions  import collect_list

def als_grid_search_ranking(train,validation,maxIters,regParams, ranks,spark):
    '''
        train: train set
        val: validation set 
        maxIters: list of maxIter
        regParams: list of regParams
        ranks: list ofranks
        spark: Spark Object
    '''
    models = {} # dict of models_parameter:ALS_obj
    precision_at_k_scores = {} 
    maps ={} #mean avearage precision
    NDCGs = {} 
    times = {} #time to run differrent models 
    rmses = {}

    sc = SparkContext.getOrCreate()
    validation.createOrReplaceTempView('validation')

    # grid-search
    for r in ranks:
        for Iter in maxIters:
            for reg in regParams:
                st = time()
                # initialize and train model 
                model = ALS(rank = r,maxIter=Iter, regParam=reg,userCol='userId',\
                            itemCol='movieId',ratingCol='rating',seed = 5,coldStartStrategy='drop',nonnegative=True)
                model = model.fit(train)
                models[(r,Iter,reg)] = model
                    
                #RMSE
                evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
                predictions=model.transform(validation)
                rmse=evaluator.evaluate(predictions)


                # evaluate on validation 
                preds = model.recommendForAllUsers(500)
                preds.createOrReplaceTempView('preds')
                
                val = spark.sql('SELECT userId, movieId FROM validation SORT BY rating DESC')
                val = val.groupBy('userId').agg(collect_list('movieId').alias('movieId_val'))
                val.createOrReplaceTempView('val')
                predAndTruth = spark.sql('SELECT preds.recommendations, val.movieId_val FROM val join preds on preds.userId = val.userId')
                predAndTruth = predAndTruth.collect()
                final_predAndTruth = []
                for item in predAndTruth:
                    truth = item[1]
                    pred = [i.movieId for i in item[0]]
                    final_predAndTruth += [(pred,truth)]
                    
                    
                final_predAndTruth =  sc.parallelize(final_predAndTruth)
                
                ranking_obj = RankingMetrics(final_predAndTruth)
                precision_at_k_scores[(r,Iter,reg)] = ranking_obj.precisionAt(500)
                maps[(r,Iter,reg)] = ranking_obj.meanAveragePrecision
                NDCGs[(r,Iter,reg)] = ranking_obj.ndcgAt(500)
                rmses[(r,Iter,reg)] = rmse
                times[(r,Iter,reg)] = round(time() - st,5)
                print('-----------------------------------------------------------------------------')
                print('For Model with maxIter = {}, reg = {}, rank = {}'.format(Iter,reg,r))
                print('Precision:',precision_at_k_scores[(r,Iter,reg)],'MAP:',maps[(r,Iter,reg)],'NDCGs:',NDCGs[(r,Iter,reg)],'RMSE:',rmse,'Time Taken:',times[(r,Iter,reg)])
                print('-----------------------------------------------------------------------------')
    return models, precision_at_k_scores,maps, NDCGs,rmses,times


def main(spark, netID):
    train_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/train_ratings_small.parquet')
    val_ratings_small = spark.read.parquet(f'hdfs:/user/{netID}/ml-latest-small/val_ratings_small.parquet')
    maxIters = [5,10,15]
    regParams = [0.1, 0.01]
    ranks = [4,5,6,7,8,9,10,11,12]
    models, precision_at_k_scores,maps, NDCGs,rmses,times = als_grid_search_ranking(train_ratings_small,val_ratings_small,maxIters,regParams, ranks,spark)

    print("SMALL DATASET:")
    print('**********************************************************************************************')
    print(precision_at_k_scores)
    print('**********************************************************************************************')
    print(maps)
    print('**********************************************************************************************')
    print(NDCGs)
    print('**********************************************************************************************')
    print(rmses)
    print('**********************************************************************************************')
    print(times)
    print('**********************************************************************************************')


if __name__ == '__main__':

    # Create the spark session object
    spark = SparkSession.builder.appName('Split_Data').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)


