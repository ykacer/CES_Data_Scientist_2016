import hadoopy
import os
import sys
import happybase
import numpy as np

hdfs_path = 'simplewikiFromHbase'
hdfs_output = 'indexwikiFromMapReduce'

def main():
    if hadoopy.exists(hdfs_output):
        hadoopy.rmr("-skipTrash %s"%hdfs_output)
    hadoopy.launch(hdfs_path,hdfs_output,'WordCount.py',files=['../stop_words.txt'])
    #word_counts = dict(hadoopy.readtb(hdfs_output))
    #for word in word_counts:
    #    print word
    #    print word_counts[word]

if __name__ == '__main__':
    main()
