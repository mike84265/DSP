#!/bin/bash
ITER=$1;
if [ -z $ITER ]; then
    echo "$0 <iter>"
    exit 1;
fi

./train $ITER model_init.txt ../seq_model_01.txt model_01.txt 
./train $ITER model_init.txt ../seq_model_02.txt model_02.txt 
./train $ITER model_init.txt ../seq_model_03.txt model_03.txt 
./train $ITER model_init.txt ../seq_model_04.txt model_04.txt 
./train $ITER model_init.txt ../seq_model_05.txt model_05.txt 
