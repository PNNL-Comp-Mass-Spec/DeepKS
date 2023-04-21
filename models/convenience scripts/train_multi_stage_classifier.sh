if [[ ${PWD##*/} != "convenience scripts" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/convenience scripts";
else
    cd ../../.. && python3 -m DeepKS.models.multi_stage_classifier --load /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/bin/deepks_nn_weights.1.cornichon --device cpu --test /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/data/raw_data_6406_formatted_95_5616.csv -c
fi
