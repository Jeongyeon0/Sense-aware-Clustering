#!/bin/bash


num_k=k_1
fold=clsuter

for name in average
do
    for sim in 05 06
    do
        python3 make_sense2cluster_tag.py $fold/$num_k/cls_result/$name/cos_$sim/ $fold/$num_k/tag_mapping/$name/cos_$sim/ $fold/$num_k/dict/ann.$name.cluster$sim.tag $fold/$num_k/dict/ann_${name}_${sim}_restrict.txt
        python3 sense2cluster.py data_result/ $fold/$num_k/tag_mapping/$name/cos_$sim/ $fold/$num_k/conll/$name/cos_$sim/
        python3 gen_tag_recover.py $fold/$num_k/tag_mapping/$name/cos_$sim/ $fold/$num_k/dict/ann.$name.${sim}_dict.tag

    done
done



