#!/usr/bin/env python

import hadoopy


class mapper(object):
    def __init__(self):
        self.d = 0.25
        generator_vector = hadoopy.readtb("hdfs://localhost:9000/user/user/vector")
        self.r = {}
        for score in generator_vector:
            self.r[score[0].encode('utf-8')] = score[1]
    def map(self,key, value):
        for v in value:
            product = (1-self.d)*1/len(value)*self.r[key]
            yield v,product

class reducer(object):
    def __init__(self):
        self.d = 0.25
        generator_vector = hadoopy.readtb("hdfs://localhost:9000/user/user/vector")
        self.r = {}
        for score in generator_vector:
            self.r[score[0].encode('utf-8')] = score[1]
    def reduce(self,key, values):
        summation = 0
        for v in values:
            summation = summation+v
        summation = summation + self.d*self.r[key]
        yield key.encode('utf-8'),summation 
  
if __name__ == "__main__":
    hadoopy.run(mapper, reducer, doc=__doc__)
