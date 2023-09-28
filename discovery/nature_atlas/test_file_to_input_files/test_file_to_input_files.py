import pandas as pd, pickle

from ....models.multi_stage_classifier import MultiStageClassifier
from ....config.join_first import join_first
from ....models.DeepKS_evaluation import eval_and_roc_workflow


def main():
    with open(join_first("bin/deepks_msc_weights.1.cornichon", 3, __file__), "rb") as f:
        msc: MultiStageClassifier = pickle.load(f)

    test_filename = join_first("data/raw_data_6500_formatted_95_5698.csv", 3, __file__)
    resave_loc = join_first("bin/deepks_msc_weights.1.resaved.cornichon", 3, __file__)
    eval_and_roc_workflow(multi_stage_classifier=msc, test_filename=test_filename, resave_loc=resave_loc)


if __name__ == "__main__":
    main()
