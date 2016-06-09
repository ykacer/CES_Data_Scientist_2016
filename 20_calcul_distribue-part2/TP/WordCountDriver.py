#!/usr/bin/env python

import hadoopy

input_path = "/alice.txt"
output_path = "/result"

if hadoopy.exists(output_path):
    hadoopy.rmr("-skipTrash %s"%output_path)
    
hadoopy.launch(input_path, output_path, 'WordCount.py')

word_counts = dict(hadoopy.readtb(output_path))

for word in word_counts:
    print "%s: %d" % (word,word_counts[word])
