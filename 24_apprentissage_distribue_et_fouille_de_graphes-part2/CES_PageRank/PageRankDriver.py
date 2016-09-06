

import hadoopy
import numpy as np
import os

input_path = "hdfs://localhost:9000/user/user/edge_list.tb"
output_path = "hdfs://localhost:9000/user/user/vector"
temp_path = "hdfs://localhost:9000/user/user/temp"

def read_vector(vect):
    for i,v in enumerate(vect):
        yield str(i).encode('utf-8'),v

def read_scores(scores):
    for s in scores:
        yield s,scores[s]

N = 64375
d = 0.5

diff=1.

r0 = np.ones(N).astype(np.float)/N

if hadoopy.exists(output_path):
    hadoopy.rmr("-skipTrash %s"%output_path)
hadoopy.writetb(output_path,read_vector(r0))

if hadoopy.exists(temp_path):
    hadoopy.rmr("-skipTrash %s"%temp_path)

while diff>0.01:
    if hadoopy.exists(temp_path):
        generator_vector = hadoopy.readtb("hdfs://localhost:9000/user/user/vector")
        generator_vector_pp = hadoopy.readtb("hdfs://localhost:9000/user/user/temp/part-00000")
        scores = {}
        for score in generator_vector:
            scores[score[0].encode('utf-8')] = d*score[1]
        for score in generator_vector_pp:
            scores[score[0].encode('utf-8')] = score[1]
        hadoopy.rmr("-skipTrash %s"%output_path)
        hadoopy.writetb(output_path,read_scores(scores))
        hadoopy.rmr("-skipTrash %s"%temp_path)
    hadoopy.launch(input_path,temp_path,'PageRank.py',files=[])

    generator_vector = hadoopy.readtb("hdfs://localhost:9000/user/user/vector")
    rk = {}
    for score in generator_vector:
        rk[score[0]] = score[1]

    generator_vector = hadoopy.readtb("hdfs://localhost:9000/user/user/temp/part-00000")
    rkpp = {}
    for score in generator_vector:
        rkpp[score[0]] = score[1]

    diff = 0.0
    for r in rkpp.keys():
        diff = diff + np.abs(rkpp[r]-rk[r])
 
    print diff

os.system('hdfs dfs -cp -f '+temp_path+'/part-00000 '+output_path)
scores = dict(hadoopy.readtb(output_path))
max_label = max(scores, key=scores.get)

labels = {}
i = 0
with open("labels") as file_object:
    while True:
        line = file_object.readline()
        if not line:
            break
        labels[str(i).encode('utf-8')] = line
        i = i+1

print "most important article of simple wikipedia : "+labels[max_label]
