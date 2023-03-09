from ..tools.make_call_graph import DeepKSCallGraph
from ..models.individual_classifiers import main

DeepKSCallGraph().make_call_graph(
    main,
    [
        "--train",
        "/root/ML/DeepKS_/DeepKS/data/raw_data_31834_formatted_65_26610.csv",
        "--val",
        "/root/ML/DeepKS_/DeepKS/data/raw_data_6406_formatted_95_5616.csv",
        "--device",
        "cuda:4",
        "-s",
    ],
)
