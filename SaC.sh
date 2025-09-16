
k=1
result=new_SDHC_allow_synonym_result


for mode in average centroid
do
    fold_name=cos_05
    threshold=0.5
    for file_name in noun verb adj adv
    do
        python3 SaC.py --vector_file SimCSE_vector/$file_name.vector --output $result/k_$k/$mode/$fold_name/$file_name.cls.result --nbit -1 --threshold $threshold --mode $mode --num_k 1
    done

    fold_name=cos_06
    threshold=0.6

    for file_name in noun verb adj adv
    do
        python3 SaC.py --vector_file SimCSE_vector/$file_name.vector --output $result/k_$k/$mode/$fold_name/$file_name.cls.result --nbit -1 --threshold $threshold --mode $mode --num_k 1
    done


    fold_name=cos_07
    threshold=0.7

    for file_name in noun verb adj adv
    do
        python3 SaC.py --vector_file SimCSE_vector/$file_name.vector --output $result/k_$k/$mode/$fold_name/$file_name.cls.result --nbit -1 --threshold $threshold --mode $mode --num_k 1
    done


    fold_name=cos_08
    threshold=0.8

    for file_name in noun verb adj adv
    do
        python3 SaC.py --vector_file SimCSE_vector/$file_name.vector --output $result/k_$k/$mode/$fold_name/$file_name.cls.result --nbit -1 --threshold $threshold --mode $mode --num_k 1
    done


    fold_name=cos_09
    threshold=0.9

    for file_name in noun verb adj adv
    do
        python3 SaC.py --vector_file SimCSE_vector/$file_name.vector --output $result/k_$k/$mode/$fold_name/$file_name.cls.result --nbit -1 --threshold $threshold --mode $mode --num_k 1
    done
done






