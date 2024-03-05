#-*-coding:utf-8 -*-
import os
import sys
from simcse import SimCSE


def main():
    #kor_sentence = ['나는 오늘 학교에 갔다', '나는 오늘 학교에서 공부를 한다']
    data_path = sys.argv[1]
    file_list = sorted(os.listdir(data_path))
#model = SentenceTransformer('bert-large-nli-mean-tokens')
    model = SimCSE('princeton-nlp/sup-simcse-roberta-large')
    print('work!!!!!') 
    for filename in file_list:
        print(filename)
        gloss = open(data_path+filename,'r',encoding='utf-8').readlines()
        save = open(sys.argv[2]+filename+'.vector','w',encoding='utf-8')
        sentences=[]
        senses = []
        for line in gloss:
            line = line.strip()
            line = line.split('\t')
            senses.append(line[0])
            sentences.append(line[-1])
        # Compute embeddings.

        embeddings = model.encode(sentences)
        cnt = 0
        for sense, vector in zip(senses, embeddings.tolist()):
            cnt+=1
            save.write(sense)
            for k in range(len(vector)):
                save.write(' '+str(vector[k]))
            save.write('\n')
            if cnt%1000==0:
                print(cnt)

    '''
        print(sentences)
        for i in range(len(sentences)):
            print(sentences[i])
            embedding = model.encode(sentences[i])
            save.write(sense[i])
            print(len(embedding))

            for k in range(len(embedding[0])):
                save.write(' '+str(embedding[0][k]))
            save.write('\n')
            if i%1000==0:
                print(i)
        
    '''
    '''
    save.write(kor_sentence[i])
    for k in range(len(embedding)):
        save.write(' '+str(embedding[k]))
    save.write('\n')
    '''
    '''
    # Compute similarity matrix. Higher score indicates greater similarity.
    df = pd.DataFrame(kor_result)
    distMatrix = pdist(df,metric='euclidean')
    print(squareform(distMatrix))
    '''

if __name__=='__main__':
    main()
