from pyspark.ml.recommendation import ALS

from pyspark.sql import Row

import elly_func
import json


def spark_session(name='session'):
    from pyspark.sql import SparkSession
    ss = SparkSession.builder.appName(name).config(
        'spark.executor.memory', '4g').getOrCreate()
    return ss


def main():
    # time_start = time.time()
    path_voc = '../resource/asnlib/publicdata/'
    train_data = path_voc + 'train_review.json'  # sys.argv[1]

    sc = elly_func.start_spark('train')
    textRDD = sc.textFile(train_data).map(elly_func.tojson)

    # 91730
    user_num = textRDD.map(lambda key: (key['user_id'])).distinct().count()
    user = textRDD.map(lambda key: (key['user_id'])).distinct().collect()

    # Build Dictionaries.
    user_dic = {}
    for i in range(user_num):
        user_dic[user[i]] = i

    # 13167 bus
    bus_num = textRDD.map(lambda key: (key['business_id'])).distinct().count()
    bus = textRDD.map(lambda key: (key['business_id'])).distinct().collect()

    bus_dic = {}
    for i in range(bus_num):
        bus_dic[bus[i]] = i

    def reverseDict(dic_in: dict):
        result = dict(map(lambda item: (item[1], item[0]), dic_in.items()))
        return result

    # Dictionaries Broadcast
    user_dic_r = reverseDict(user_dic)
    bus_dic_r = reverseDict(bus_dic)

    user_dic = sc.broadcast(user_dic)
    bus_dic = sc.broadcast(bus_dic)
    user_dic_r = sc.broadcast(user_dic_r)
    bus_dic_r = sc.broadcast(bus_dic_r)

    # Create DataFrame for Training.
    pairs = textRDD.map(lambda key: (user_dic.value[key['user_id']], bus_dic.value[key['business_id']], key['stars']))
    ratingsRDD = pairs.map(lambda p: Row(user_id=int(p[0]), bus_id=int(p[1]), rating=float(p[2])))
    ratings = spark_session(ratingsRDD).createDataFrame(ratingsRDD)

    # Firstly remove cold starts.
    def mapping(key):
        try:
            return (user_dic.value[key['user_id']], bus_dic.value[key['business_id']])
        except KeyError:
            return (-1, -1)

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to NaN
    print("Training...")
    als = ALS(maxIter=8, regParam=0.3, rank=100, seed=424, userCol="user_id", itemCol="bus_id", ratingCol="rating",
              coldStartStrategy="nan")
    model = als.fit(ratings)
    print('Saving ALS...')
    model.write().overwrite().save('elly_model')

    print('Saving dics...')

    with open('user_dic.json', 'w') as f:
        json.dump(user_dic.value, f)
    with open('user_dic_r.json', 'w') as f:
        json.dump(user_dic_r.value, f)
    with open('bus_dic.json', 'w') as f:
        json.dump(bus_dic.value, f)
    with open('bus_dic_r.json', 'w') as f:
        json.dump(bus_dic_r.value, f)


if __name__ == "__main__":
    main()
