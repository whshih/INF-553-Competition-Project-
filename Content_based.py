import csv
import elly_func
import re
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StopWordsRemover
import time
import json

"""
Spark DataFrame 的庫
https://spark.apache.org/docs/2.1.0/ml-features.html
CountVectorizer VS ****HashingTF 比較
https://towardsdatascience.com/countvectorizer-hashingtf-e66f169e2d4e
 含TFIDF + 最高頻200字 + 向量化
+--------------------+--------------------+--------------------+--------------------+
|         business_id|                text|            filtered|            features|
+--------------------+--------------------+--------------------+--------------------+
|3xykzfVY2PbdjKCRD...|[don, sushi, but,...|[sushi, husband, ...|(200,[0,1,2,3,4,5...|
|R7-Art-yi73tWaRTu...|[daughter, had, h...|[daughter, fillin...|(200,[1,3,4,5,6,7...|
|DYuOxkW4DtlJsTHdx...|[came, here, with...|[came, one, worke...|(200,[0,1,2,3,4,5...|
|EZ4TljJvGenxrkM4J...|[this, place, yum...|[place, yummy, or...|(200,[0,1,2,3,4,5...|
|cJWbbvGmyhFiBpG_5...|[have, been, here...|[least, times, br...|(200,[0,1,2,3,4,5...|
|EsVVTcYO9DhSK2oxa...|[was, very, excit...|[excited, try, pl...|(200,[0,1,2,3,4,5...|
|KqxJcRmPEdiUTXHws...|[was, our, first,...|[first, visit, an...|(200,[0,1,2,3,4,5...|
|DkYS3arLOhA8si5uU...|[this, place, ope...|[place, open, whe...|(200,[0,1,2,3,4,5...|
|I-KFzdnJcqbUHOAdO...|[service, friendl...|[service, friendl...|(200,[0,1,2,3,4,5...|
|p5rpYtxS5xPQjt3MX...|[love, this, plac...|[love, place, nic...|(200,[0,1,2,3,4,5...|
|gEjHZH--OgwXph03p...|[very, fun, poker...|[fun, poker, room...|(200,[0,1,2,3,4,5...|
|bipVXy6DVI0xJaXsc...|[had, the, pleasu...|[pleasure, hiring...|(200,[0,1,2,3,4,5...|
|KIWUFC4iSjUH4v4t_...|[have, been, goin...|[going, kathy, ye...|(200,[0,1,3,4,5,6...|
|3_QNAH8yVzY0sBPft...|[walked, without,...|[walked, without,...|(200,[0,1,2,3,4,5...|
|pwdZMC9Q2QH2uNbUj...|[amazing, broth, ...|[amazing, broth, ...|(200,[0,1,2,3,4,5...|
|6K59UtSTXt56F4rMo...|[purchased, new, ...|[purchased, new, ...|(200,[0,1,3,4,5,6...|
|xvdDcG7oWT_KxU9fn...|[very, reasonably...|[reasonably, pric...|(200,[0,1,2,3,4,5...|
|b24sTA2JkHB4joNlN...|[fries, are, extr...|[fries, extremely...|(200,[0,1,2,3,4,5...|
|Z4I8ebf8_VKQavfsl...|[took, our, car, ...|[took, car, chevr...|(200,[0,1,3,4,5,6...|
|XZbuPXdyA0ZtTu3Az...|[pork, tenderloin...|[pork, tenderloin...|(200,[0,1,2,3,4,5...|
+--------------------+--------------------+--------------------+--------------------+
"""
def spark_session(name = 'session'):
    from pyspark.sql import SparkSession
    ss = SparkSession.builder.appName(name).config(
        'spark.executor.memory', '4g').getOrCreate()
    return ss


def remove_blank(x):
    words = re.findall("[a-zA-Z]+", x)
    result = []
    for i in words:
        if len(i) > 2:
            result.append(i.lower())
    return result

def main():
    time_start = time.time()
    data = 'train_review.json'  # sys.argv[1]
    sc = elly_func.start_spark('Final_Project')

    # total pairs = 1029758
    textRDD = sc.textFile(data).map(elly_func.tojson).map(
        lambda x: ((x['user_id'], x['business_id']), x['text'])).reduceByKey(
        lambda a, b: a + b).mapValues(remove_blank).map(lambda x: (x[0][0], x[0][1], x[1]))

    # Create DataFrame
    tableA = spark_session(textRDD).createDataFrame(textRDD, ['user_id', 'business_id', 'text'])

    # Remove stopwords
    remover = StopWordsRemover(inputCol="text", outputCol="filtered")

    # text不用了 拉掉
    df = remover.transform(tableA).drop('text')

    # 轉換成tokenizer要的形式 string
    test = df.withColumn("sentence", df["filtered"].cast("string"))

    # 將句子篩選有用字詞
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(test)

    # TF-IDF 高頻200字篩選
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=200)
    featurizedData = hashingTF.transform(wordsData)

    # 字詞向量化
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    # user與business與字詞關係 按字邏輯排列
    user_profile = rescaledData.select('user_id', 'business_id', 'features').orderBy("user_id")
    business_profile = rescaledData.select('business_id', 'features').orderBy("business_id")


    def set2list(x):
        temp = []
        for i in x:
            temp.append(i)
        return temp

    # user與business與字詞關係 字典建立

    # 91730
    # Total number of [('business_id', ['case','eat',...]),...]
    business_dic = business_profile.rdd.map(lambda x: (x[0], list(x[1].indices))).reduceByKey(
        lambda a, b: a + b).mapValues(lambda x: set2list(set(x))).collectAsMap()

    # 13167
    user_dic = user_profile.rdd.map(lambda x: (x[0], list(x[2].indices))).reduceByKey(lambda a, b: a + b).mapValues(
        lambda x: set2list(set(x))).collectAsMap()
    # user_profile_dic = {'user1': [word2, word8, word24,....],.....}
    user_bus = sc.textFile(data).map(elly_func.tojson).map(
        lambda x: ((x['user_id'], x['business_id']))).reduceByKey(lambda a, b: a + b).collectAsMap()

#   差cos sim
    time_end = time.time()
    print('Duration:', time_end - time_start)
if __name__ == "__main__":
    main()