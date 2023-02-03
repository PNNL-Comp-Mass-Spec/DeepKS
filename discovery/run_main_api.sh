if [[ ${PWD##*/} != "discovery" ]] ;
then 
    echo "Failed: please run this script from DeepKS/discovery/";
else
    python3 -m DeepKS.api.main --dry-run --scores -p in_order_json -v --cartesian-product --device cpu -kf discovery/kinase_list.txt -sf discovery/site_list.txt
fi