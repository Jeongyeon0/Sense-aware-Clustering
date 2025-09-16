#-*-coding:utf-8-*-
import sys
import os

def remove_sense(sense):
    word = sense.split('%')[0]
    return word


def main():
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    file_list = os.listdir(data_path)
    #file_list = ['noun.relation.vector.cluster_result.txt']
    restrict_dict={}
    restrict = open(sys.argv[4],'w',encoding='utf-8')
    tag_name = open(sys.argv[3],'w',encoding='utf-8')
    for name in file_list:
        f = open(data_path+name,'r',encoding='utf-8').readlines()
        temp = name.split('.')
        save = open(save_path+temp[0]+'.'+temp[1]+'.cluster.tag','w', encoding='utf-8')
        for line in f:
            line = line.strip()
            line = line.split(' ')
            cluster_name = temp[0].replace('_','.')+':'+temp[1]+':'+line[0]
            tag_name.write(cluster_name+'\n')
            sense_list = line[1:]
            for sense in sense_list:
                word = remove_sense(sense)
                if word in restrict_dict.keys():
                    restrict_dict[word].append(cluster_name)
                else:
                    restrict_dict[word]=[]
                    restrict_dict[word].append(cluster_name)

                save.write(sense+'\t'+cluster_name+'\n')

    sorted_list = sorted(restrict_dict.keys())
    for word in sorted_list:
        restrict.write(word)
        for i in range(len(restrict_dict[word])):
            restrict.write(' '+restrict_dict[word][i])
        restrict.write('\n')


if __name__=='__main__':
    main()
