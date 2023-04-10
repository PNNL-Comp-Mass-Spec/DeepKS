#! /bin/zsh

if [[ ${PWD##*/} != "convenience scripts" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/convenience scripts";
else
    cd ../../../
    python3 -m DeepKS.models.individual_classifiers --train data/raw_data_31834_formatted_65_26610.csv --val data/raw_data_6500_formatted_95_5698.csv --device cuda:4 -s 
fi
