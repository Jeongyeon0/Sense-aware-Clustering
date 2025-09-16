#!/bin/sh

mode=$1
epoch=$2
pred_path=new_ckpt/semcor_lr_2e-5_with_warmup2000/$mode/$epoch

for name in senseval2 senseval3 semeval2007 semeval2013 semeval2015
do
    echo $name
    java Scorer ../senseval/corpus/$name/original/$name.gold.key.txt $pred_path/$name.pred
done
