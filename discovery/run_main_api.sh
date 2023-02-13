#! /bin/zsh

if [[ ${PWD##*/} != "discovery" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/";
else
    SITE_SIZE=29914
    KIN_SIZE=492
    cd ../../
    # stdbuf -o0 python3 -m trace --ignore-dir /opt/conda/envs/py310/lib --trace --module 
    python3 -m DeepKS.api.main --suppress-seqs-in-output --scores --normalize --groups -p dictionary_json --cartesian-product --device cuda:6 -kf discovery/kinase_list_$KIN_SIZE.txt -sf discovery/site_list_$SITE_SIZE.txt --kin-info discovery/compact_kinase_info_$KIN_SIZE.json --site-info discovery/compact_site_info_$SITE_SIZE.json

fi