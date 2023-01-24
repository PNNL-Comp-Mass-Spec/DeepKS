if [[ ${PWD##*/} != "tests" ]] ;
then 
    echo "Failed: please run this script from DeepKS/tests/";
else
    cd ../..

    python3 -m DeepKS.api.main -kf tests/sample_inputs/kins.txt -s VQQEPGWTCYLFSYV,NHSVNQHWANFTCNR,ALVVNQRDKSYNAQA -p in_order

    python3 -m DeepKS.api.main -kf tests/sample_inputs/kins.txt -sf tests/sample_inputs/sites.txt -p in_order -v

    python3 -m DeepKS.api.main -k TCHKGIDKMMRMQHAMLPLQMYLCF,YVMLYNNGPLWGRNDMMSCKSYVHD,HHMCEFCCAMCPQDGWHLMTAFGHD -sf tests/sample_inputs/sites.txt -p in_order

fi