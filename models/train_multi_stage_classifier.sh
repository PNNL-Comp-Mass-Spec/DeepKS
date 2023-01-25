if [[ ${PWD##*/} != "models" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/";
else
    cd ../.. && python3 -m DeepKS.models.multi_stage_classifier --load /Users/druc594/Library/CloudStorage/OneDrive-PNNL/Documents/DeepKS_/DeepKS/bin/indivudial_classifiers_2023-01-13T19:22:36.1738012023-01-13T19:22:36.173809.pkl --test ../data/raw_data_6406_formatted_95_5616.csv --device cpu -c
fi
