#! /bin/zsh

if [[ ${PWD##*/} != "discovery" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/";
else
    cd ~/Desktop/
    # stdbuf -o0 python3 -m trace --ignore-dir /opt/conda/envs/py310/lib --trace --module 
    python3 -m DeepKS.api.main --scores --groups -p dictionary_json --cartesian-product --device cpu -kf discovery/kinase_list_tiny.txt -sf discovery/site_list_tiny.txt
fi