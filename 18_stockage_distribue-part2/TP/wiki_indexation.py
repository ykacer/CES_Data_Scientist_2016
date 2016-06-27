import sys
import happybase
import re
from stemming.porter2 import stem
import numpy as np

connection = happybase.Connection('localhost')

if 'wiki' not in connection.tables():
    sys.exit("Error : no wiki table found")
else:
    print "OK : wiki table found"
table_wiki = connection.table('wiki')

if 'indexWiki' in connection.tables():
    connection.delete_table('indexWiki',True)
connection.create_table('indexWiki',{'wiki':{}})
table_index = connection.table('indexWiki')

# download stop words :
words_stop = [line.rstrip('\n') for line in open('stop_words.txt')]

NdocsPerWord = {}

NdocsMax = 30000
for i,(key,data) in enumerate(table_wiki.scan()):
    url = key.decode('utf-8')
    print "currently indexing url : ",url
    # get text from url :
    text = data['cf:body'].decode('utf-8')
    # tokenize text :
    words_it = re.split('\W+',text)
    # stem words :
    words_stemmed = [stem(word.lower()) for word in words_it if word not in words_stop and word not in '']
    # remove redundancy :
    words_stemmed_unique = list(set(words_stemmed))
    # number of words in document :
    nT = len(words_stemmed_unique)
    # compute tf for each word of document :
    batch = table_index.batch()
    for word in words_stemmed_unique:
        if word in NdocsPerWord:
            NdocsPerWord[word] += 1 
        else:
            NdocsPerWord[word] = 1
        nt = words_stemmed.count(word)
        tf = 1.0*nt/nT
        # print "-- ",word,":",nt,",",tf
        batch.put(word.encode('utf-8'),{"wiki:"+url:str(tf).encode('utf-8'),"wiki:url":url.encode('utf-8')})
    batch.send()
    print "tf : ",tf
    if i+1>=NdocsMax:
        break;

Ndocs = i
print "Total number of documents : ",Ndocs

batch = table_index.batch()
for key,data in table_index.scan():
    word = key.decode('utf-8')
    url = data['wiki:url'].decode('utf-8')
    print "currently indexing word : ",word
    tf = float(data['wiki:'+url].decode('utf-8'))
    idf = np.log10(1.0*Ndocs/NdocsPerWord[word])
    tfidf = tf*idf
    print "tfidf : ",tfidf
    batch.put(word.encode('utf-8'),{"wiki:"+url:str(tfidf).encode('utf-8')})
    batch.delete(word.encode('utf-8'),{"wiki:url"})
batch.send()
