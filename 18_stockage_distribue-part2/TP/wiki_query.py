import sys
import happybase
import re
from stemming.porter2 import stem
import numpy as np

connection = happybase.Connection('localhost')

if 'indexWiki' not in connection.tables():
    sys.exit("Error : no indexWiki table found")
else:
    print "OK : indexWiki table found"
    table_index = connection.table('indexWiki')

words_stop = [line.rstrip('\n') for line in open('stop_words.txt')]

#query = "Which associations are both Singapore and Brunei in?"
query = raw_input('Enter your query :')
words_query = [stem(word.group(0)).lower() for word in
        re.finditer(r"\w+",query,re.UNICODE) if word.group(0).lower() not in words_stop]

documents_intersection = set.intersection(*[set(table_index.row(word).viewkeys()) for word in words_query]) 

scores = np.zeros(len(documents_intersection))
for word in words_query:
    data = table_index.row(word)
    for i,doc in enumerate(documents_intersection):
        tfidf = float(data[doc].decode('utf-8'))
        scores[i] = scores[i]+tfidf

arg_sort = np.argsort(scores)
arg_sort[:] = arg_sort[::-1]
documents_intersection = list(documents_intersection)
print "top-5 ranking : "
for k,i in enumerate(arg_sort):
    doc = documents_intersection[i][5:]
    print('http://localhost/articles/%s/%s/%s/%s.html (score :%0.3f)' % (doc[0].lower(),doc[1].lower(),doc[2].lower(),doc,scores[i]))
    if k>=4:
        break
