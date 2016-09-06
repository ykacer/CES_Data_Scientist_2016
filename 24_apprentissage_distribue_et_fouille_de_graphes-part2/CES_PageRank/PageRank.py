#!/usr/bin/env python

import hadoopy


class mapper(object):
    def __init__(self):
        self.d = 0.25
    def map(self,key, value):
        url = key[0].encode('utf-8')
        r = key[1]
        for v in value:
            product = 1.0/len(value)*r
            yield v.encode('utf-8'),product
        yield url,value

class reducer(object):
    def __init__(self):
        self.d = 0.25
        self.N = 64375
    def reduce(self,key, values):
        links = []
        r = 0
        for v in values:
            if type(v) is list: 
                links = v
            else:
                r = r + v

        r = (1-self.d)*r + self.d/self.N
        yield (key.encode('utf-8'),r),links
  
if __name__ == "__main__":
    hadoopy.run(mapper, reducer, doc=__doc__)
