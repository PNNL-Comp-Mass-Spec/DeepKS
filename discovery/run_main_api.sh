#! /bin/zsh

if [[ ${PWD##*/} != "discovery" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/";
else
    cd ../../
    # stdbuf -o0 python3 -m trace --ignore-dir /opt/conda/envs/py310/lib --trace --module 
    python3 -m DeepKS.api.main --scores --normalize --groups -p dictionary_json --cartesian-product --device cpu -kf discovery/kinase_list_20.txt -sf discovery/site_list_50.txt --kin-info discovery/compact_kinase_info_20.json --site-info discovery/compact_site_info_57.json


fi