import sys
import os

def main():

    dic = {}
    file_list = os.listdir(sys.argv[1])
    for name in file_list:
        f = open(sys.argv[1]+name,'r').readlines()
        for line in f:
            line = line.strip()
            line = line.split()
            dic[line[0]]=line[1]

    sorted_key = sorted(dic.keys())
    save = open(sys.argv[2],'w')
    for key in sorted_key:
        save.write(key+'\t'+dic[key]+'\n')





if __name__=='__main__':
    main()
