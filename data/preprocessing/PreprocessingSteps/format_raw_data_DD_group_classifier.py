import re, json, random, pathlib, os, pandas as pd
from typing import Union

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)



from .... import config

mode = config.cfg.get_mode() 
random.seed(0)


def my_pop(df, index):
    """
    Based on https://stackoverflow.com/a/58548326
    """
    row = df.iloc[index]
    df.drop(index, inplace=True)
    return row


def generate_real(input_fn, kin_seq_file, grp_fam_kin_file, config):
    try:
        assert "held_out_percentile" in config
        assert "train_percentile" in config
        held_out_percentile = config["held_out_percentile"]
        train_percentile = config["train_percentile"]
        pre_split_or_post_split = config["pre_split_or_post_split"]
    except AssertionError as ae:
        print(
            "Missing one of the following keys in config: `held_out_percentile`, `train_percentile`,"
            " `num_held_out_kins`, `pre_split_or_post_split`."
        )
        raise ae

    all_data = (
        pd.read_csv(input_fn)
        .sort_values(by=["num_sites", "lab", "seq"], ascending=[False, True, True])
        .reset_index(drop=True)
        .applymap(lambda x: x.upper() if isinstance(x, str) else x)
    )
    if pre_split_or_post_split == "pre":
        for ML_set, percentile in [
            ("../te_kins.json", held_out_percentile),
            ("../vl_kins.json", held_out_percentile),
            ("../tr_kins.json", train_percentile),
        ]:
            kins = [x.replace("(", "").replace(")", "").replace("*", "") for x in json.load(open(ML_set, "r"))]
            all_df = (
                pd.merge(
                    pd.DataFrame({"lab_updated": kins, "uniprot_id": [x.split("|")[1] for x in kins]}),
                    all_data,
                    how="left",
                    on="uniprot_id",
                )
                .drop("lab", axis=1, inplace=False)
                .rename(columns={"lab_updated": "lab"}, inplace=False)
            )
            generate_real_helper(all_df, input_fn, kin_seq_file, grp_fam_kin_file, percentile)

    elif pre_split_or_post_split == "post":
        raise RuntimeError("Not Implemented.")
    else:
        raise RuntimeError("`pre_split_or_post_split` is not one of `pre` or `post`.")


def generate_real_helper(
    section_df: pd.DataFrame,
    input_fn: str,
    kin_seq_file: str,
    grp_fam_kin_file: str,
    percentile: Union[int, float],
    write_file=True,
):
    kin_to_seq = pd.read_csv(kin_seq_file, sep="\t")
    kin_to_seq["Symbol"] = (
        kin_to_seq["gene_name"].apply(lambda x: re.sub(r"[\(\)\*]", "", x).upper()) + "|" + kin_to_seq["kinase"]
    )

    kin_to_grp = pd.read_csv(grp_fam_kin_file)
    kin_to_grp["Symbol"] = (
        kin_to_grp["Kinase"].apply(lambda x: re.sub(r"[\(\)\*]", "", x).upper()) + "|" + kin_to_grp["Uniprot"]
    )

    all_df = (
        pd.merge(section_df, kin_to_seq, how="left", left_on="lab", right_on="Symbol")
        .merge(kin_to_grp, how="left", on="Symbol")[["Symbol", "Group", "organism", "kinase_seq", "num_sites"]]
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    grp_to_grp_ind = {g: j for j, g in enumerate(sorted(set(all_df["Group"].values)))}
    all_df["Class"] = all_df["Group"].apply(lambda x: grp_to_grp_ind[x])

    extra = "" if mode == "no_alin" else "_alin"
    all_df.rename(
        inplace=True, columns={"organism": "Organism", "kinase_seq": "Kinase Sequence", "num_sites": "Num Sites"}
    )
    all_df = all_df.sort_values(by=["Class", "Organism", "Symbol", "Kinase Sequence"])
    all_df = all_df[["Symbol", "Organism", "Group", "Class", "Num Sites", "Kinase Sequence"]]
    fn = "None"
    if write_file:
        all_df.to_csv(
            fn := re.sub("([0-9]+)", f"{len(all_df)}", input_fn).replace(".csv", "")
            + "_group_classifier"
            + f"_formatted{extra}.csv",
            index=False,
        )
    print(f"Outputted: {fn}")


if __name__ == "__main__":
    if mode == "no_alin":
        kin_seq_file = "../../raw_data/kinase_seq_822.txt"
    else:
        kin_seq_file = "../../raw_data/kinase_seq_alin_792.txt"

    generate_real(
        "../../raw_data/raw_data_22473.csv",
        kin_seq_file,
        "../kin_to_fam_to_grp_817.csv",
        {"held_out_percentile": 95, "train_percentile": 65, "num_held_out_kins": 60, "pre_split_or_post_split": "pre"},
    )
