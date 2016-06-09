#!/usr/bin/env python

import hadoopy
import re

def mapper(key, value):
    for word in re.split("[^a-z]+",value.lower()):
        if word:
            yield word, 1

def reducer(key, values):
    accum = 0
    for count in values:
        accum += int(count)
    if(accum>100):
        yield key, accum

if __name__ == "__main__":
    hadoopy.run(mapper, reducer, reducer, doc=__doc__)