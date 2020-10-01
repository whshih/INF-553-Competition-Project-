import json
import sys
import csv

def spark_session(name = 'session'):
    from pyspark.sql import SparkSession
    ss = SparkSession.builder.appName(name).config(
        'spark.executor.memory', '2g').getOrCreate()
    return ss
def start_spark(name = 'SparkTask', local = 'local[*]'):
    from pyspark import SparkContext
    from pyspark import SparkConf

    confSpark = SparkConf().setAppName(name).setMaster(
        local).set('spark.driver.memory', '4G').set(
            'spark.executor.memory', '4G')
    sc = SparkContext.getOrCreate(confSpark)
    sc.setLogLevel(logLevel='ERROR')
    return sc


def tojson(line):
    lines = line.splitlines()
    data = json.loads(lines[0])
    return data


def list2dic(list):
    dic = {}
# Break the collection of user_id and count in to user_id dictionary --> dictionary={ row num: 'user_id'}
# The num of user_id = 26184
    for i in range(len(list)):
        if list[i][0] in dic:
            break
        else:
            dic[list[i][0]] = i
    return dic

def save_model(data1, data2, file_path):
    import pickle
    with open(file_path, 'wb') as model:
        pickle.dump(data1, model, pickle.HIGHEST_PROTOCOL)
        pickle.dump(data2, model, pickle.HIGHEST_PROTOCOL)



