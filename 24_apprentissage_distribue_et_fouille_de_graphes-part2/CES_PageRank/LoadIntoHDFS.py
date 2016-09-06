import hadoopy

tb_path="hdfs://localhost:9000/user/user/edge_list.tb"

if hadoopy.exists(tb_path):
    hadoopy.rmr("-skipTrash %s"%tb_path)

def read_edge_wiki(file_object):
    while True:
        line = file_object.readline().split()
        if not line:
            break
        yield line[0].decode('utf-8'),[l.decode('utf-8') for l in line[1:]]
        #yield line[0].decode('utf-8'),line[1].decode('utf-8')

def main():
    with open('edge_list.txt') as f:
        hadoopy.writetb(tb_path,read_edge_wiki(f))

if __name__ == '__main__':
    main()

