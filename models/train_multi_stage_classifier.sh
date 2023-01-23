if cd ../.. ;
then 
    echo "";
else
    echo "Failed to change directory to DeepKS/models/../..";
    echo "Please run this script from DeepKS/models/";
fi
python3 -m DeepKS.models.multi_stage_classifier --load /people/druc594/ML/DeepKS/bin/saved_state_dicts/indivudial_classifiers_2023-01-11T19:47:44.8914632023-01-11T19:47:44.891470.pkl --test ../data/raw_data_6406_formatted_95_5616.csv --device cuda:6 -c