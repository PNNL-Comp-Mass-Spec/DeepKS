if [[ ${PWD##*/} != "models" ]] ;
then 
    echo "Failed: please run this script from DeepKS/models/";
else
    cd ../.. && python3 -m DeepKS.models.multi_stage_classifier --load /home/dockeruser/DeepKS/bin/deepks_nn_weights.1.cornichon --device cpu -c --test /home/dockeruser/DeepKS/tests/sample_inputs/small_val_or_test.csv
fi
