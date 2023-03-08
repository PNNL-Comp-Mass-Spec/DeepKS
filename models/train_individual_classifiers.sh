#! /bin/zsh

if [[ ${PWD##*/} != "models" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/";
else
    KIN_SIZE=10
    SITE_SIZE=86201
    cd ../../
    python3 -m DeepKS.models.individual_classifiers --train /root/ML/DeepKS_/DeepKS/data/raw_data_31834_formatted_65_26610.csv --val /root/ML/DeepKS_/DeepKS/data/raw_data_6406_formatted_95_5616.csv --device cuda:4 -s 
fi
