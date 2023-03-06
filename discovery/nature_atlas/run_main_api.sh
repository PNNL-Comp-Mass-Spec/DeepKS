#! /bin/zsh

if [[ ${PWD##*/} != "nature_atlas" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/nature_atlas";
else
    KIN_SIZE=10
    SITE_SIZE=89784
    cd ../../../
    python3 -m DeepKS.api.main --suppress-seqs-in-output --scores --normalize --groups -p csv --cartesian-product --device cpu -kf discovery/nature_atlas/kinase_list_$KIN_SIZE.txt -sf discovery/nature_atlas/site_list_$SITE_SIZE.txt --kin-info discovery/nature_atlas/compact_kinase_info_$KIN_SIZE.json --site-info discovery/nature_atlas/compact_site_info_86201.json --bypass-group-classifier
fi
