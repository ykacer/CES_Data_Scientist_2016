import hadoopy
import os
import sys
import happybase
import numpy as np

hdfs_index = 'indexWikiFromMapReduce'

connection = happybase.Connection('localhost','9090')

if 'indexWikiFromMapReduce' in connection.tables():
    connection.delete_table('indexWikiFromMapReduce',True)
connection.create_table('indexWikiFromMapReduce',{'wiki':{}})
table_index = connection.table('indexWikiFromMapReduce')


def main():
    word_counts = dict(hadoopy.readtb(hdfs_index))
    for word in word_counts:
        batch = table_index.batch()
        for url in word_counts[word]:
            tfidf = word_counts[word][url]
            batch.put(word.encode('utf-8'),{"wiki:"+url:str(tfidf).encode('utf-8')})
        batch.send()
if __name__ == '__main__':
    main()
