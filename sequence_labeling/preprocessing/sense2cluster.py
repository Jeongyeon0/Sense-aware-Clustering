#-*-coding:utf-8-*-
import sys
import os


def main():
    pos_file = open('symbol.tag','r',encoding='utf-8').readlines()
    pos = []
    for line in pos_file:
        line = line.strip()
        pos.append(line)
    print(pos)
    data_path = sys.argv[1]
    mapping_path = sys.argv[2]
    save_path = sys.argv[3]
    tag_list = os.listdir(mapping_path)
    cluster_dict={}
    cluster_dict['00']='00'
    for name in tag_list:
        f = open(mapping_path+name,'r',encoding='utf-8').readlines()
        for line in f:
            line = line.strip()
            line = line.split('\t')
            cluster_dict[line[0]]=line[1]
    corpus_list = os.listdir(data_path)
    for name in corpus_list:
        f = open(data_path+name,'r',encoding='utf-8').readlines()
        save = open(save_path+name.replace('.sense','.conll'),'w',encoding='utf-8')
        for line in f:
            if line=='\n':
                save.write('\n')
            else:
                line = line.replace(' ','_').strip()
                line = line.split('\t')
                if line[3] in pos:
                    save.write(line[0]+'\t'+line[1] + '\t' + cluster_dict[line[2]].replace('_','.') + '\t' + '.' + '\n')
                else:
                    save.write(line[0]+'\t'+line[1] + '\t' + cluster_dict[line[2]].replace('_','.') + '\t' + line[3] + '\n')



if __name__=='__main__':
    main()
