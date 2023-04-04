#! /bin/zsh

if [[ ${PWD##*/} != "api" ]] ;
then 
    echo "Failed: please run this script from DeepKS/api";
else
    export UN="dockeruser"
    cd ../../
        python3 -m DeepKS.api.main -kf /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery/nature_atlas/issue28B/kins.txt -sf /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery/nature_atlas/issue28B/sites.txt -p csv --kin-info /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery/nature_atlas/issue28B/kin-info.json --site-info /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery/nature_atlas/issue28B/site-info.json --scores --cartesian-product --groups --device cpu --pre-trained-nn /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/bin/deepks_nn_weights.-1.cornichon --pre-trained-gc /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/bin/deepks_gc_weights.6.cornichon --normalize-scores --bypass-group-classifier
fi