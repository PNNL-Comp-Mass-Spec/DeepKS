#! /bin/zsh

if [[ ${PWD##*/} != "nature_atlas" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/nature_atlas";
else
    KIN_SIZE=10
    SITE_SIZE=86201
    cd ../../../
    python3 -m DeepKS.api.main --suppress-seqs-in-output --scores --normalize --cartesian-product --groups -p csv --device cpu -kf discovery/nature_atlas/kinase_list_$KIN_SIZE.txt -sf discovery/nature_atlas/site_list_$SITE_SIZE.txt --kin-info discovery/nature_atlas/compact_kinase_info_$KIN_SIZE.json --site-info discovery/nature_atlas/compact_site_info_$SITE_SIZE.json --pre-trained-msc bin/deepks_msc_weights_resaved_11.cornichon 
fi
