#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
import re
from stemming.porter2 import stem
import numpy as np
import happybase

words_stop = [line.rstrip('\n') for line in open('../stop_words.txt')]
words_stop.append('')

sc=SparkContext()

hbaseConfig={"hbase.mapreduce.inputtable":"wiki","hbase.mapreduce.scan.columns":"cf:body"}
table_rdd=sc.newAPIHadoopRDD("org.apache.hadoop.hbase.mapreduce.TableInputFormat","org.apache.hadoop.hbase.io.ImmutableBytesWritable","org.apache.hadoop.hbase.client.Result",keyConverter="org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter",valueConverter="org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter",conf=hbaseConfig)
#for (key,data) in table_rdd.takeSample(True,3):
#    print "--------------url:"
#    print key.decode('utf-8')
#    print "---------------text:"
#    print data.decode('utf-8')

splitText = table_rdd.map(lambda (url,text):(url,[stem(word.group().lower()) for word in re.finditer(r"\w+",text,re.UNICODE) if word.group().lower() not in words_stop]))

tf = splitText.map(lambda (url,splittedText):(url,{word:1.0*splittedText.count(word)/len(splittedText) for word in splittedText}))

tfWordAsKey = tf.flatMap(lambda (url,tf):[(word,[(url,tf[word])])for word in tf]).reduceByKey(lambda a,b:a+b)

tfidf = tfWordAsKey.map(lambda (word,tfList):(word,[(url,tf*np.log10(27474.0/len(tfList))) for (url,tf) in tfList]))

#A = tfidf.takeSample(True,5)
#print A

connection = happybase.Connection('localhost','9090')

if 'indexwikiFromSpark2' in connection.tables():
    connection.delete_table('indexwikiFromSpark2',True)
    connection.create_table('indexwikiFromSpark2',{'wiki':{}})
    table_index = connection.table('indexwikiFromSpark2')

tfidf.foreachpartition((word,data) => table_index.put(word.encode('utf-8'),{"wiki:"+url:str(tfidf).encode('utf-8')}) for url,tfidf in data)



