#! /bin/zsh

if [[ ${PWD##*/} != "convenience scripts" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/convenience scripts";
else
    cd ../../../
    python3 \
    -m DeepKS.models.individual_classifiers \
    --train data/raw_data_31834_formatted_65_26610.csv \
    --val data/raw_data_6500_formatted_95_5698.csv \
    --device cuda:4 \
    --pre-trained-gc bin/deepks_gc_weights.-1.cornichon \
    --ksr-params models/hyperparameters/KSR_params_GRU.json \
    --groups NON-TK
fi
