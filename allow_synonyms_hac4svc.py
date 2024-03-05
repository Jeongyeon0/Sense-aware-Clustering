import sys
import faiss
import numpy as np
from numpy.linalg import norm
import argparse
import time

np.random.seed(1314)


def euclidean_dist(A, B):
    A = np.array(A).astype('float32')
    B = np.array(B).astype('float32')
    return norm(A - B)
    # return np.sqrt(np.sum((A-B)**2))


def cos_sim(A, B):
    A = np.array(A).astype('float32')
    B = np.array(B).astype('float32')
    return np.dot(A, B) / (norm(A) * norm(B))
    # return (1+(np.dot(A, B)/(norm(A)*norm(B))))/2


def remove_tag(target_list):
    word_list = []
    for target in target_list:
        word = target.split('%')[0]
        word_list.append(word)
    return word_list


def isHomonym_conflict(cluster1, cluster2):
    cluster1 = set(remove_tag(cluster1))
    cluster2 = set(remove_tag(cluster2))
    if len(cluster1 & cluster2) == 0:
        return False
    else:
        return True


def lazy_sampling(cluster1, cluster2):
    pass


def group_average_distance(cls_dict, vec_dict, query, target):
    dist = 0.0
    for i in range(len(cls_dict[query])):
        for j in range(len(cls_dict[target])):
            dist += cos_sim(vec_dict[cls_dict[query][i]], vec_dict[cls_dict[target][j]])
    dist = dist / (len(cls_dict[query]) * len(cls_dict[target]))
    return dist


def single_distance(cls_dict, vec_dict, query, target):
    min_dist = 1.0
    for i in range(len(cls_dict[query])):
        for j in range(len(cls_dict[target])):
            dist = cos_sim(vec_dict[cls_dict[query][i]], vec_dict[cls_dict[target][j]])
            if dist <= min_dist:
                min_dist = dist
    return min_dist


def complete_distance(cls_dict, vec_dict, query, target):
    max_dist = -1.0
    for i in range(len(cls_dict[query])):
        for j in range(len(cls_dict[target])):
            dist = cos_sim(vec_dict[cls_dict[query][i]], vec_dict[cls_dict[target][j]])
            if dist >= max_dist:
                max_dist = dist
    return max_dist


def clustering_and_get_centroid(cls_dict, centroid_vec_dict, target):
    maintain_target = target[0]
    remove_target = target[1]

    centroid_vec_dict[maintain_target] *= len(cls_dict[maintain_target])
    centroid_vec_dict[remove_target] *= len(cls_dict[remove_target])
    centroid_vec_dict[maintain_target] += centroid_vec_dict[remove_target]
    cls_dict[maintain_target].extend(cls_dict[remove_target])

    centroid_vec_dict[maintain_target] /= len(cls_dict[maintain_target])
    del centroid_vec_dict[remove_target]

    del cls_dict[remove_target]
    return cls_dict, centroid_vec_dict


def return_total_sense_num(cls_dict):
    # 클러스터링 검토용 함수
    n = 0
    for key in cls_dict.keys():
        n += len(cls_dict[key])
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_file", type=str, default="noun.animal.vec")
    parser.add_argument("--nbit", type=int, default=-1)
    parser.add_argument("--output_file", type=str, default="test_result2.txt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_k", type=int, default=1)
    parser.add_argument("--mode", type=str, default="centroid", choices=["single", "complete", "centroid", "average"])

    arg = parser.parse_args()

    # File open
    f1 = open(arg.vector_file, 'r').readlines()

    # Defining dictionaries used in this program
    cls_dict = {}
    centroid_vec_dict = {}
    vec_dict = {}
    complete_cls_dict = {}

    print(
        "Parameter setting\n nbit: {0}, threshold: {1}, num_k: {2}, mode: {3}\n input_file: {4}, output_file: {5} ".format(
            arg.nbit, arg.threshold, arg.num_k, arg.mode, arg.vector_file, arg.output_file))

    # Loading vectors from input file
    print('Now load vector file...')
    # Note that initial data type is float64. However, when you want to use the other function such as vector normalization, you should change the data type as float32.


    for i, line in enumerate(f1):
        line = line.strip()
        line = line.split()
        sense = line[0]
        vec = line[1:]
        centroid_vec_dict[i] = np.array(vec).astype('float32')
        cls_dict[i] = [sense]
        vec_dict[sense] = vec

    # Parametert setting
    threshold = arg.threshold
    if arg.nbit == -1:
        nbit = int(np.sqrt(len(cls_dict.keys())))
    else:
        nbit = arg.nbit
    print('Construct LSH maps...')
    # Constructing LSH(Locality Sensitive Hashing) function by using faiss library and input data
    centroid_vec_list = np.array(list(centroid_vec_dict.values())).astype('float32')
    index = faiss.IndexLSH(centroid_vec_list.shape[1], nbit)
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(centroid_vec_list, np.array(list(cls_dict.keys())).astype('int64'))

    query_cand = list(cls_dict.keys())
    flag = 0
    cnt = 0
    total_len = len(query_cand)
    print('Start clustering...')
    start = time.time()
    while (1):
        # query = np.random.choice(len(centroid_vec_list), 1, False)
        if flag == 0 or flag==1:
            query = np.random.choice(len(query_cand), 1, False)
            query_key = list(cls_dict.keys())[query[0]]
            index.remove_ids(np.array([query_key]))

        elif flag == 2:
            query_cand.remove(query_key)
            query = np.random.choice(len(query_cand), 1, False)
            query_key = list(query_cand)[query[0]]
            index.remove_ids(np.array([query_key]))

        # Defining exit condition
        if len(query_cand) <= 1:
            break
        # Clustering steps
        distance, I = index.search(centroid_vec_dict[query_key].reshape(1, -1), k=arg.num_k)
        for d, i in zip(distance[0], I[0]):
            # 모호성 없는 데이터가 입력으로 사용되면 가능 => 테스트용 종료조건
            if i == -1:
                break

            # Distance calculation
            if arg.mode == 'centroid':
                cosine_value = cos_sim(centroid_vec_dict[query_key], centroid_vec_dict[i])
            elif arg.mode == 'average':
                cosine_value = group_average_distance(cls_dict, vec_dict, query_key, i)
            elif arg.mode == 'single':
                cosine_value = single_distance(cls_dict, vec_dict, query_key, i)
            elif arg.mode == 'complete':
                cosine_value = complete_distance(cls_dict, vec_dict, query_key, i)
            else:
                print("Please check initial argument!")
                exit()

            # print(cosine_value)
            if cosine_value >= threshold:
                if (isHomonym_conflict(cls_dict[query_key], cls_dict[i]) == False) or (cosine_value==1.0):
                    cls_dict, centroid_vec_dict = clustering_and_get_centroid(cls_dict, centroid_vec_dict, [query_key, i])
                    index.remove_ids(np.array([i]))
                    query_cand = list(cls_dict.keys())
                    flag = 0
                else:
                    complete_cls_dict[query_key]=cls_dict[query_key]
                    complete_cls_dict[i]=cls_dict[i]
                    index.remove_ids(np.array([i]))

                    del centroid_vec_dict[query_key]
                    del centroid_vec_dict[i]
                    del cls_dict[query_key]
                    del cls_dict[i]
                    query_cand = list(cls_dict.keys())
                    flag=1
                break
            else:
                flag =2

        # print(centroid_vec_dict[query_key],query_key)
        if flag!=1:
            index.add_with_ids(np.array(centroid_vec_dict[query_key]).reshape(1, -1).astype('float32'),
                           np.array([query_key]).astype('int64'))
        # print('now cls_dict: {0}, now flag: {1}'.format(cls_dict.keys(),flag))

        if cnt % 500 == 0:
            dict_size = len(cls_dict.keys())
            print('now step: {0}, clustering state: {1}/{2}, compression_rate: {3: .3f}%'.format(cnt, dict_size,
                                                                                                 total_len, (1 - (
                            dict_size / total_len)) * 100))
        cnt += 1

    for key in cls_dict.keys():
        complete_cls_dict[key] = cls_dict[key]

    end = time.time()
    print('save the clustering results...')
    save = open(arg.output_file, 'w')
    # save = open('test_result.txt','w')


    for key in complete_cls_dict.keys():
        save.write(str(key))
        for i in range(len(complete_cls_dict[key])):
            save.write(' ' + complete_cls_dict[key][i])
        save.write('\n')

    print('time spend: {}'.format(end - start))


if __name__ == '__main__':
    main()
