import sys
import os
import time
import numpy as np
import faiss
from scipy.spatial.distance import squareform


def _prepare_out_argument(out, dtype, expected_shape):
    if out is None:
        return np.empty(expected_shape, dtype=dtype)
    if out.shape != expected_shape:
        raise ValueError("Output array has incorrect shape.")
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous.")
    if out.dtype != np.double:
        raise ValueError("Output array must be double type.")
    return


def _pdist_callable(X, cluster_dict, *, out=None, **kwargs):
    n = len(X)
    out_size = (n * (n - 1)) // 2
    dm = _prepare_out_argument(out, np.float32, (out_size,))
    index_dm = _prepare_out_argument(out, np.int, (out_size, 2))
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if isHomonym(cluster_dict[i], cluster_dict[j]) == True:
                dm[k] = -1.0  # 동음이의어 위치값을 -1로 표시
                index_dm[k] = [i, j]
                k += 1
                continue
            dist = 0
            for z_i in range(len(X[i])):
                for z_j in range(len(X[j])):
                    dist += np.linalg.norm(
                        np.array(X[i][z_i], dtype=np.float32) - np.array(X[j][z_j], dtype=np.float32))
            dm[k] = dist / (len(X[i]) * len(X[j]))
            index_dm[k] = [i, j]
            k += 1
    return dm, index_dm


def _pdist(X, cluster_dict):
    return _pdist_callable(X, cluster_dict)


def remove_tag(target_list):
    word_list = []
    for target in target_list:
        word = target.split('%')[0]
        word_list.append(word)
    return word_list


def get_centroid(cluster1, cluster2, vec_dict, cluster_dict):

    pass

# 동음이의어가 있는지 검사
def isHomonym(cluster1, cluster2):
    cluster1 = set(remove_tag(cluster1))
    cluster2 = set(remove_tag(cluster2))
    if len(cluster1 & cluster2) == 0:
        return False
    else:
        return True


def _clustering(sim_index, vec, target_index):
    vec[sim_index[target_index][0]].extend(vec[sim_index[target_index][1]])
    del (vec[sim_index[target_index][1]])
    return vec


def _synonym_clustering(sim_index, vec, synonym_index, cluster_dict):
    del_index = []
    for i in range(len(synonym_index[0])):
        index = synonym_index[0][-i - 1]
        vec[sim_index[index][0]].extend(vec[sim_index[index][1]])
        cluster_dict[sim_index[index][0]].extend(cluster_dict[sim_index[index][1]])
        vec[sim_index[index][1]] = []
        cluster_dict[sim_index[index][1]] = []
    return vec, cluster_dict


def _update(sim_matrix, sim_index, target_index, X, max_dist, cluster_dict):
    update_index = sim_index[target_index][0]
    del_index = sim_index[target_index][1]

    for k in range(len(sim_index)):
        if update_index in sim_index[k]:
            dist = 0.0
            if isHomonym(cluster_dict[sim_index[k][0]], cluster_dict[sim_index[k][1]]) == True:
                sim_matrix[k] = -1.0
                continue
            for i in range(len(X[sim_index[k][0]])):
                for j in range(len(X[sim_index[k][1]])):
                    dist += np.linalg.norm(
                        np.array(X[sim_index[k][0]][i], dtype=np.float32) - np.array(X[sim_index[k][1]][j],
                                                                                     dtype=np.float32))
            sim_matrix[k] = dist * (1 / (len(X[sim_index[k][0]]) + len(X[sim_index[k][1]]))) * (1 / max_dist)

        if del_index in sim_index[k]:
            sim_matrix[k] = -1.0

    return sim_matrix


def main():
    start = time.time()
    file_list = os.listdir(sys.argv[1])
    for name in file_list:

        print(name)
        # print('data loading......')
        data = open(sys.argv[1] + name, 'r').readlines()
        cluster_dict = []
        vec = []
        vec_index = [0] * len(data)
        for i, lines in enumerate(data):
            line = lines.strip()
            line = line.split(' ')
            cluster_dict.append([line[0]])
            vec.append([line[1:]])
            vec_index[i] = i
        if len(cluster_dict) == 1:
            save = open(sys.argv[2] + name.replace('.vector', '.sep_cls.txt'), 'w')
            save.write('0\t' + cluster_dict[0][0])
            continue
        sim_matrix, sim_index = _pdist(vec, cluster_dict)
        max_dist = max(sim_matrix)
        '''
        주의사항 : 모든 vector 사이의 거리를 미리 계산해둔다면 좀 더 빠른 속도로 클러스터링 할 수 있지만, 메모리 문제로 불가능
        '''

        '''
        동음이의어인 경우, 클러스터링 대상에서 제외
        '''
        try:
            zero_index = np.where(sim_matrix == -1.0)
            sim_matrix = np.delete(sim_matrix, zero_index[0])
            sim_index = np.delete(sim_index, zero_index[0], axis=0)

            '''
            동의어를 일괄적으로 클러스터링
            '''
            synonym_index = np.where(sim_matrix == 0.0)
            vec, cluster_dict = _synonym_clustering(sim_index, vec, synonym_index, cluster_dict)
            sim_matrix = np.delete(sim_matrix, synonym_index[0])
            sim_index = np.delete(sim_index, synonym_index[0], axis=0)

            sim_matrix = sim_matrix * (1 / max_dist)
            target_index = np.argmin(sim_matrix)

            for threshold in [1.000000]:
                while (sim_matrix[target_index] <= threshold):
                    ''' 
                    원활한 인덱스 접근을 위해 vector를 저장한 리스트의 인덱스는 변하지 않음.
                    ex) (0, 2)를 클러스터링 -> cluster_dict[0]과 vector[0]에 cluster_dict[2]와 vector[2]의 모든 값을 옮김. cluster_dict[2]와 vector[2]는 빈 상태로 유지하여 index에 변화를 주지않아 접근을 원활하게 만듦 
                    '''
                    vec[sim_index[target_index][0]].extend(vec[sim_index[target_index][1]])
                    vec[sim_index[target_index][1]] = []
                    cluster_dict[sim_index[target_index][0]].extend(cluster_dict[sim_index[target_index][1]])
                    cluster_dict[sim_index[target_index][1]] = []

                    sim_matrix = _update(sim_matrix, sim_index, target_index, vec, max_dist, cluster_dict)

                    zero_index = np.where(sim_matrix == -1.0)
                    sim_matrix = np.delete(sim_matrix, zero_index[0])
                    sim_index = np.delete(sim_index, zero_index[0], axis=0)

                    target_index = np.argmin(sim_matrix)

                save = open(sys.argv[2] + name.replace('.vector', '.sep_cls.txt'), 'w')
                for i in range(len(cluster_dict)):
                    if len(cluster_dict[i]) == 0:
                        continue

                    save.write(str(i) + '\t')
                    for j in range(len(cluster_dict[i])):
                        if j == 0:
                            save.write(cluster_dict[i][j])
                        else:
                            save.write(' ' + cluster_dict[i][j])
                    save.write('\n')

        except:
            save = open(sys.argv[2] + name.replace('.vector', '.sep_cls.txt'), 'w')
            for i in range(len(cluster_dict)):
                if len(cluster_dict[i]) == 0:
                    continue
                save.write(str(i) + '\t')
                for j in range(len(cluster_dict[i])):
                    if j == 0:
                        save.write(cluster_dict[i][j])
                    else:
                        save.write(' ' + cluster_dict[i][j])
                save.write('\n')


if __name__ == '__main__':
    main()
