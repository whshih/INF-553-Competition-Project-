import json
from sys import argv

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import Row
import elly_func

data2 = argv[1]  # '../CompetitionProject/dataset/test_review.json'  #
output_file = argv[2]  # 'out.json'  #


def spark_session(name='session'):
    from pyspark.sql import SparkSession
    ss = SparkSession.builder.appName(name).config(
        'spark.executor.memory', '4g').getOrCreate()
    return ss


def mapping(key):
    try:
        return (user_dic.value[key['user_id']], bus_dic.value[key['business_id']])
    except KeyError:
        return (-1, -1)


def cold_stars(stars, user, bus, user_dic, bus_dic, avg):
    try:
        return bus_dic[bus]
    except KeyError:
        try:
            return user_dic[user]
        except KeyError:
            return avg


def mapping_r(key):
    return (user_dic_r.value[str(key[1])], bus_dic_r.value[str(key[0])], key[2])


# Starting sc and ss
sc = elly_func.start_spark('predict', local = 'local[3]')
ss = spark_session('Pred')

# Loading Models
model = ALSModel.load('elly_model')

with open('user_dic.json', 'r') as f:
    user_dic = sc.broadcast(json.load(f))
with open('user_dic_r.json', 'r') as f:
    user_dic_r = sc.broadcast(json.load(f))
with open('bus_dic.json', 'r') as f:
    bus_dic = sc.broadcast(json.load(f))
with open('bus_dic_r.json', 'r') as f:
    bus_dic_r = sc.broadcast(json.load(f))

path_voc = '../resource/asnlib/publicdata/'
user_stars_mean_path = path_voc + 'user_avg.json'
bus_stars_mean_path = path_voc + 'business_avg.json'
with open(user_stars_mean_path, 'r') as f:
    user_stars_mean = sc.broadcast(json.load(f))
with open(bus_stars_mean_path, 'r') as f:
    bus_stars_mean = sc.broadcast(json.load(f))

textRDD2 = sc.textFile(data2).map(elly_func.tojson)

# Load our test data.
test = textRDD2.map(mapping).filter(lambda x: 1 if x[0] != -1 else 0)
testRDD = test.map(lambda p: Row(user_id=int(p[0]), bus_id=int(p[1])))
test = spark_session(testRDD).createDataFrame(testRDD)  # .show()

predictions = model.transform(test).rdd.map(mapping_r)


def cold(key):
    try:
        return (user_dic.value[key['user_id']], bus_dic.value[key['business_id']], -1)
    except KeyError:
        return (key['user_id'], key['business_id'], 1)


avgRatings = 3.7999  # ratings.select('rating').groupBy().avg().first()[0]

# Give the rating for the prediction of cold stars.
cold_start = textRDD2.map(cold).filter(lambda line: 1 if line[2] != -1 else 0).map(
    lambda x: (x[0], x[1], cold_stars(x[2], x[0], x[1], user_stars_mean.value, bus_stars_mean.value, avgRatings)))

output = cold_start.union(predictions).collect()

with open(output_file, 'w') as f:
    for x in output:
        json.dump({'user_id': x[0], 'business_id': x[1], 'stars': x[2]}, f)
        f.write('\n')
