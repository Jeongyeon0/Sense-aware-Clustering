# elements

CUDA_VISIBLE_DEVICES=0 python3 new_trainer.py --batch_size 16 --lr 2e-5 --n_epochs 20 --finetuning --logdir "new_ckpt/semcor_lr_2e-5_with_warmup2000/average_05" --train --trainset "new_ann_average_05_conll/semcor_corpus.conll" --validset "new_ann_average_05_conll/" --restrict_file "ann_average_05_restrict.txt" --recover_file "ann.average.05_dict.tag" --sense_tag_file "ann.average.cluster05.tag"  --pos_tag_file "general_pos.index"

CUDA_VISIBLE_DEVICES=0 python3 new_trainer.py --batch_size 16 --lr 2e-5 --n_epochs 20 --finetuning --logdir "new_ckpt/semcor_lr_2e-5_with_warmup2000/average_06" --train --trainset "new_ann_average_06_conll/semcor_corpus.conll" --validset "new_ann_average_06_conll/" --restrict_file "ann_average_06_restrict.txt" --recover_file "ann.average.06_dict.tag" --sense_tag_file "ann.average.cluster06.tag"  --pos_tag_file "general_pos.index"
