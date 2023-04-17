#! /bin/zsh

if [[ ${PWD##*/} != "api" ]]; then
    echo "Failed: please run this script from DeepKS/api"
else
    export UN="dockeruser"
    cd ../../
    python3 -m DeepKS.api.main -kf discovery/nature_atlas/issue28B/kins.txt -sf discovery/nature_atlas/issue28B/sites.txt -p csv --kin-info discovery/nature_atlas/issue28B/kin-info.json --scores --cartesian-product --groups --device cpu --pre-trained-nn bin/deepks_nn_weights.0.cornichon --bypass-group-classifier --pre-trained-gc /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/bin/deepks_gc_weights.0.cornichon
fi
