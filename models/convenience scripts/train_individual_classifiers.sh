#! /bin/zsh

if [[ ${PWD##*/} != "convenience scripts" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/convenience scripts";
else
    export UN="dockeruser"
    cd ../../../
    python3 -m DeepKS.models.individual_classifiers --train /home/$UN/DeepKS/data/raw_data/raw_data_45176_formatted_65.csv --val /home/$UN/DeepKS/data/raw_data_6500_formatted_95_5698.csv --device cuda:4 -s 
fi
