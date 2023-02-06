if [[ ${PWD##*/} != "models" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/";
else
    cd ../.. && python3 -m DeepKS.models.multi_stage_classifier --load /root/ML/DeepKS_/DeepKS/bin/indivudial_classifiers_2023-02-03T18:43:30.314486.pkl --device cuda:4 -c
fi
