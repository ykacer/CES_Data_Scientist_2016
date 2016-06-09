#!/usr/bin/env python

from pyspark import SparkContext
import re

input_path="/alice.txt"

sc=SparkContext()

lines = sc.textFile(input_path)
words_occ=lines.flatMap(lambda l:re.split("[^a-z]+",l.lower())).filter(lambda l:l)
words=words_occ.map(lambda l:(l,1)).reduceByKey(lambda a,b:a+b).filter(lambda (a,b):b>100)
word_counts=dict(words.collectAsMap())

for word in word_counts:
    print "%s: %d" % (word,word_counts[word])
