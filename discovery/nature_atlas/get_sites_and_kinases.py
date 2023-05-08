# %% IMPORTS ---
import pandas as pd, sys, pathlib, os, re, collections, random, warnings
from typing import Union

try:
    from ..discovery_preparation import format_kin_and_site_lists

    os.chdir(pathlib.Path(__file__).parent.resolve())
except NameError:
    # Jupyter Notebook, Probably
    sys.path.append("/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery")
    from discovery_preparation import format_kin_and_site_lists

    os.chdir("/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery/nature_atlas/")


def main():
    # %% CONSTANTS ---
    NUM_KIN_ASSERT_AVAIL = 303
    NUM_RANDOM_KINS = 10
    SEED = 42

    # %% PROCESSING KINASES ---
    kin_seq_file = "../../data/raw_data/kinase_seq_918.csv"
    uni_to_seq = pd.read_csv(kin_seq_file).set_index("kinase").to_dict()["kinase_seq"]
    metadata_df = pd.read_csv("./41586_2022_5575_MOESM3_ESM.csv")
    matrix_name_to_uniprot_id: dict[str, str] = metadata_df.set_index("Matrix_name").to_dict()["Uniprot id"]
    uniprot_id_to_known_group: dict = metadata_df.set_index("Uniprot id").to_dict()[
        "Family"
    ]  # Atlas paper calls kinase group "Family".

    set_of_ok_group_names = {"<UNANNOTATED>", "ATYPICAL", "AGC", "CAMK", "CK1", "CMGC", "OTHER", "STE", "TK", "TKL"}

    nature_map = {"Other": "OTHER", "FAM20": "ATYPICAL", "PIKK": "ATYPICAL", "PDHK": "ATYPICAL", "Alpha": "ATYPICAL"}

    uniprot_id_to_known_group = {}
    for k, v in uniprot_id_to_known_group.items():
        if v in set_of_ok_group_names:
            uniprot_id_to_known_group[k] = v
        elif v in nature_map:
            uniprot_id_to_known_group[k] = nature_map[v]
        else:
            uniprot_id_to_known_group[k] = (None, v)

    for k, v in uniprot_id_to_known_group.items():
        if isinstance(v, tuple) and v[1] is None:
            warnings.warn(
                f'For kinase {k}, got "known" group annotation of {v}, which is not recognized. Coercing it to'
                " <UNKNOWN>."
            )
            uniprot_id_to_known_group[k] = "<UNKNOWN>"

    atlas = pd.read_csv("./41586_2022_5575_MOESM5_ESM.csv")
    available_kinases: list[str] = [str(x) for x in atlas.columns if re.search(r"^[0-9A-Z]+_percentile$", str(x))]
    assert (
        len(available_kinases) == NUM_KIN_ASSERT_AVAIL
    ), f"Expected {NUM_KIN_ASSERT_AVAIL} kinases, got {len(available_kinases)}"
    assert len(available_kinases) == len(set(available_kinases)), "Duplicate kinase names"
    random.seed(SEED)
    kin_sample_raw = random.sample(available_kinases, NUM_RANDOM_KINS)
    kins_sample = [x.split("_")[0] for x in kin_sample_raw]

    # %% GET SAMPLED KINASES ---
    sampled_kin_to_seq = {
        k: (
            uni_to_seq[matrix_name_to_uniprot_id[k]]
            if k in matrix_name_to_uniprot_id and matrix_name_to_uniprot_id[k] in uni_to_seq
            else None
        )
        for k in kins_sample
    }
    assert all([x is not None for x in sampled_kin_to_seq.values()])
    sampled_kin_to_seq = collections.OrderedDict(sorted(sampled_kin_to_seq.items(), key=lambda x: x[0]))
    sampled_kin_to_uniprot = {}
    for k in kins_sample:
        if k in matrix_name_to_uniprot_id:
            sampled_kin_to_uniprot[k] = matrix_name_to_uniprot_id[k]
        else:
            sampled_kin_to_uniprot[k] = None

    sampled_kin_to_known_group = {}
    for k in kins_sample:
        if k in matrix_name_to_uniprot_id:
            sampled_kin_to_known_group[k] = uniprot_id_to_known_group[matrix_name_to_uniprot_id[k]]
        else:
            sampled_kin_to_known_group[k] = None

    assert all([x is not None for x in sampled_kin_to_uniprot.values()])
    assert all([x is not None for x in sampled_kin_to_known_group.values()])
    sampled_kin_to_uniprot = collections.OrderedDict(sorted(sampled_kin_to_uniprot.items(), key=lambda x: x[0]))
    sampled_kin_to_known_group = collections.OrderedDict(sorted(sampled_kin_to_known_group.items(), key=lambda x: x[0]))

    # %% PROCESS SITES ---
    atlas_site_info = atlas[["Uniprot Primary Accession", "Gene", "Phosphosite", "SITE_+/-7_AA"]].copy()
    for i, r in atlas_site_info.iterrows():
        if pd.isna(r["Gene"]):
            atlas_site_info.at[i, "Gene"] = f"?UnipAc:{r['Uniprot Primary Accession']}"
    atlas_site_info["Symbol"] = atlas_site_info["Gene"] + "|" + atlas_site_info["Uniprot Primary Accession"]
    atlas_site_info["SITE_+/-7_AA"] = atlas_site_info["SITE_+/-7_AA"].apply(lambda x: x.replace("_", "X"))

    # %% FINAL STEPS AND EXPORTING TO FILES ---
    kinase_symbol_to_kinase_sequence: collections.OrderedDict[str, str] = collections.OrderedDict(
        sorted(
            {
                str(k) + "|" + str(sampled_kin_to_uniprot[k]): str(sampled_kin_to_seq[k]) for k in sampled_kin_to_seq
            }.items(),
            key=lambda x: x[0],
        )
    )

    kinase_symbol_to_known_group: collections.OrderedDict[str, str] = collections.OrderedDict(
        sorted(
            {
                str(k) + "|" + str(sampled_kin_to_uniprot[k]): str(sampled_kin_to_known_group[k])
                for k in sampled_kin_to_known_group
            }.items(),
            key=lambda x: x[0],
        )
    )

    symbol_to_location: collections.OrderedDict[str, list[str]] = collections.OrderedDict(
        sorted(atlas_site_info.groupby(by="Symbol")["Phosphosite"].apply(list).to_dict().items(), key=lambda x: x[0])
    )
    symbol_to_flanking_sites: collections.OrderedDict[str, list[str]] = collections.OrderedDict(
        sorted(atlas_site_info.groupby(by="Symbol")["SITE_+/-7_AA"].apply(list).to_dict().items(), key=lambda x: x[0])
    )

    format_kin_and_site_lists(
        kinase_symbol_to_kinase_sequence=kinase_symbol_to_kinase_sequence,
        symbol_to_location=symbol_to_location,
        symbol_to_flanking_sites=symbol_to_flanking_sites,
        save_dir=pathlib.Path(os.getcwd()).absolute(),
        kinase_symbol_to_kinase_known_group=kinase_symbol_to_known_group,
    )


if __name__ == "__main__": # pragma: no cover
    main()
