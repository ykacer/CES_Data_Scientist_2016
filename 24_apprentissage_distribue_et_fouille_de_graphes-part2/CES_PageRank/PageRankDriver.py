import hadoopy
import numpy as np
import os

edge_path = "hdfs://localhost:9000/user/user/edge_list.tb"
input_path = "hdfs://localhost:9000/user/user/input.tb"
output_path = "hdfs://localhost:9000/user/user/vector"
temp_path = "hdfs://localhost:9000/user/user/temp"

def read_vector(vect):
    for i,v in enumerate(vect):
        yield str(i).encode('utf-8'),v

N = 64375

diff=1.

r0 = np.ones(N).astype(np.float)/N

if hadoopy.exists(input_path):
    hadoopy.rmr("-skipTrash %s"%input_path)
os.system('hdfs dfs -cp '+edge_path+' '+input_path)
    
if hadoopy.exists(output_path):
    hadoopy.rmr("-skipTrash %s"%output_path)
hadoopy.writetb(output_path,read_vector(r0))

if hadoopy.exists(temp_path):
    hadoopy.rmr("-skipTrash %s"%temp_path)

iteration = 0
while diff>0.01:
    if hadoopy.exists(temp_path):
        hadoopy.rmr("-skipTrash %s"%temp_path)
    hadoopy.launch(input_path,temp_path,'PageRank.py',files=[])
    
    generator_vector = hadoopy.readtb(output_path)
    rk = {}
    for score in generator_vector:
        url = score[0]
        r = score[1]
        rk[url] = r

    generator_vector = hadoopy.readtb(temp_path+"/part-00000")
    rkpp = {}
    for i,score in enumerate(generator_vector):
        url = score[0][0]
        r = score[0][1]
        rkpp[url] = r

    diff = 0.0
    for r in rkpp.keys():
        diff = diff + np.abs(rkpp[r]-rk[r])
    print "iteration: "+str(iteration)+", error: "+str(diff)
    iteration = iteration + 1

    r = np.zeros(N)
    for i in rkpp.keys():
        r[int(i)] = rkpp[i]

    if hadoopy.exists(output_path):
        hadoopy.rmr("-skipTrash %s"%output_path)
    hadoopy.writetb(output_path,read_vector(r))

    os.system("hdfs dfs -cp -f "+temp_path+"/part-00000 "+input_path)

scores = dict(hadoopy.readtb(output_path))
max_label = max(scores, key=scores.get)

labels = {}
i = 0
with open("labels") as file_object:
    while True:
        line = file_object.readline()
        if not line:
            break
        labels[str(i).encode('utf-8')]= line
        i = i+1
print "most important article of simple wikipedia :"+labels[max_label]                                                                         
