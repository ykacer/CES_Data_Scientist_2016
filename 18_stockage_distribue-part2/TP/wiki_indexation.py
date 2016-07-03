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
    print "currently processing url : ",url
    ## get text from url :
    text = data['cf:body'].decode('utf-8')
    ## tokenize text :
    words_stemmed_unique = [stem(word.group().lower()) for word in re.finditer(r"\w+",text,re.UNICODE) if word.group().lower() not in words_stop]

    ## number of words in document :
    nT = len(words_stemmed_unique)
    ## compute tf for each word of document :
    batch = table_index.batch()
    for word in words_stemmed_unique:
        if word in NdocsPerWord:
            NdocsPerWord[word] += 1 
        else:
            NdocsPerWord[word] = 1
        nt = words_stemmed_unique.count(word)
        tf = 1.0*nt/nT
        batch.put(word.encode('utf-8'),{"wiki:"+url:str(tf).encode('utf-8')})
    batch.send()
    if i+1>=NdocsMax:
        break;

Ndocs = i
print "Total number of documents : ",Ndocs


## OBLIGER DE COUPER LE SCAN EN DEUX, SINON ERREUR...
for key,data in table_index.scan(row_stop='2240'):
    word = key.decode('utf-8')
    print "currently indexing word : ",word
    idf = np.log10(1.0*Ndocs/NdocsPerWord[word])
    for url in data.keys():
        tf = float(data[url].decode('utf-8'))
        tfidf = tf*idf
        table_index.put(word.encode('utf-8'),{url:str(tfidf).encode('utf-8')})

for key,data in table_index.scan(row_start='2240'):
    word = key.decode('utf-8')
    print "currently indexing word : ",word
    idf = np.log10(1.0*Ndocs/NdocsPerWord[word])
    for url in data.keys():
        tf = float(data[url].decode('utf-8'))
        tfidf = tf*idf
        table_index.put(word.encode('utf-8'),{url:str(tfidf).encode('utf-8')})
