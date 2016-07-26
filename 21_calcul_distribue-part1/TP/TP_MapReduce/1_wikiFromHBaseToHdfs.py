import hadoopy
import os
import sys
import happybase
import numpy as np

hdfs_path = 'wiki.seq.tb'
local_path = 'wiki.seq.tb.local'
if hadoopy.exists(hdfs_path):
    hadoopy.rmr("-skipTrash %s"%hdfs_path)

connection = happybase.Connection('localhost')

if 'wiki' not in connection.tables():
    sys.exit("Error : no wiki table found")
else:
    print "OK : wiki table found"
    table_wiki = connection.table('wiki')

NdocsMax = 30000
def read_hbase(table_hbase):
    for key,data in table_hbase.scan(limit=NdocsMax):
        yield key.decode('utf-8'),data['cf:body'].decode('utf-8')

#def read_local_dir(local_path):
#    for fn in os.listdir(local_path):
#        path = os.path.join(local_path, fn)
#        if os.path.isfile(path):
#            yield path, open(path).read()


def main():
    hadoopy.writetb(hdfs_path,read_hbase(table_wiki))
    if os.path.isfile(local_path):
        print "deleting "+local_path
        os.remove(local_path)
    os.system('hadoop fs -copyToLocal wiki.seq.tb '+local_path)

if __name__ == '__main__':
    main()
