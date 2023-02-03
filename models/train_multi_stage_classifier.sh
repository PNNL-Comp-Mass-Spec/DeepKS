if [[ ${PWD##*/} != "models" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/";
else
    cd ../.. && python3 -m DeepKS.models.multi_stage_classifier --load /root/ML/deepks-rename-trial/bin/indivudial_classifiers_2023-02-02T23:53:59.210698.pkl --device cuda:4 -c
fi
