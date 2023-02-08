#! /bin/zsh

if [[ ${PWD##*/} != "discovery" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/";
else
    cd ../../
    # stdbuf -o0 python3 -m trace --ignore-dir /opt/conda/envs/py310/lib --trace --module 
    python3 -m DeepKS.api.main --scores --suppress-seqs-in-output --normalize --groups -p dictionary_json --cartesian-product --device cuda:6 -kf discovery/kinase_list_430.txt -sf discovery/site_list_30673.txt --kin-info discovery/kinase_info_13189390.csv --site-info discovery/site_info_13189390.csv
fi