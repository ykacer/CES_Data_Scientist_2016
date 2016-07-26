#!/usr/bin/env python

from pyspark import SparkContext
import re
from stemming.porter2 import stem
import numpy as np
import hadoopy

#input_path="hdfs://localhost:9000/alice.txt"
input_hdfs_path="hdfs://localhost:9000/user/user/wiki.seq.tb"
output_hdfs_path='hdfs://localhost:9000/user/user/indewWikiFromSpark.seq.tb'

words_stop = [line.rstrip('\n') for line in open('stop_words.txt')]
words_stop.append('')

sc=SparkContext()

lines = sc.sequenceFile(input_hdfs_path).map(lambda (x,y):(x[5:].decode('utf-8'),y[5:].decode('utf-8')))

splitText = lines.map(lambda (url,text):(url,[stem(word.group().lower()) for word in re.finditer(r"\w+",text,re.UNICODE) if word.group().lower() not in words_stop]))

tf = splitText.map(lambda (url,splittedText):(url,{word:1.0*splittedText.count(word)/len(splittedText) for word in splittedText}))

tfWordAsKey = tf.flatMap(lambda (url,tf):[(word,[(url,tf[word])]) for word in tf]).reduceByKey(lambda a,b:a+b)

tfidf = tfWordAsKey.map(lambda (word,tfList):(word,[(url,tf*np.log10(27474.0/len(tfList))) for (url,tf) in tfList]))

NwordsMax = 200000
def read_rdd(rdd):
    for key,data in rdd.takeSample(True,NwordsMax):
        yield key,data

if hadoopy.exists(output_hdfs_path):
    hadoopy.rmr("-skipTrash %s"%output_hdfs_path)

hadoopy.writetb(output_hdfs_path,read_rdd(tfidf))


