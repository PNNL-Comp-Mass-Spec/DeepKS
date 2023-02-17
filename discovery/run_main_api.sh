#! /bin/zsh

if [[ ${PWD##*/} != "discovery" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/";
else
    cd /root/ML/DeepKS_/
    stdbuf -o0 python3 -m trace --ignore-dir /opt/conda/envs/py310/lib --trace --module DeepKS.api.main --scores -p in_order_json --cartesian-product --device cuda:7 -kf discovery/kinase_list_small.txt -sf discovery/site_list_small.txt
fi