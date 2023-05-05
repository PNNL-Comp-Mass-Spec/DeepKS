import re, io, pandas as pd, requests as req

HELD_OUT_FAMILY = "SGK"


from ....config.logging import get_logger
logger = get_logger()
if __name__ == "__main__":
    logger.status("Loading Modules")

def get_kin_to_fam_to_grp(relevant_kinases):
    def up(x):
        if isinstance(x, str):
            return x.upper() 
        else:
            return x
    url = "http://www.kinhub.org/kinases.html"
    r = req.request("GET", url)
    if r.status_code == 200:
        table = str(r.content)
    else:
        raise req.HTTPError(str(r.status_code) + str(r.content))
    table_io = io.StringIO(table)
    table_pd = pd.read_html(table_io)[0]
    table_pd.rename(columns={"HGNC Name\\n \\n": "Kinase"}, inplace=True)
    table_pd = table_pd[~table_pd["Manning Name\\n \\n"].str.contains("Domain2")][
        ["Kinase", "Family", "Group"]
    ]  # .to_csv('../preprocessing/kin_to_fam_to_grp.csv', index = False)
    table_pd = table_pd[~table_pd.isnull().any(axis=1)]

    table_pd = table_pd.applymap(up)
    relevant_kinases = (
        pd.read_csv(relevant_kinases)[["kinase", "gene_name"]]
        .rename({"kinase": "Uniprot", "gene_name": "Kinase"}, axis=1)
        .applymap(up)
    )

    combined_df = pd.merge(relevant_kinases, table_pd, how="left", on="Kinase")
    combined_df = combined_df[combined_df["Kinase"].isin(relevant_kinases["Kinase"])]

    additional = {
        "^PIK3[A-Z]+$": "PIK",  # Confirmed + Group
        "PIKFYVE": "PIPK",  # Confirmed + Group
        "^CDK[0-9A-Z]+$": "CDK",  # Confirmed + Group
        "^NME[0-9]$": "NDK",  # Confirmed + Group
        "^PAK[0-9]": "STE20",  # Confirmed + Group
        "^GRK[0-9]": "GRK",  # Confirmed + Group
        "^PRKA[A-Z][0-9]$": "CAMKL",  # Questionable + Group
        "^ENPP[0-9]$": "ENPP",  # Confirmed - Group unknown
        "^CSNK2.*$": "CK2",  # Questionable + Group
        "^CILK[0-9]$": "CDK",  # Confirmed + Group
        "^PGK[0-9]$": "PGK",  # Confirmed - Group unknown
        "^PCK[0-9]$": "GTP",  # Confirmed - Group unknown
        "^ALDO.*$": "FPB",  # Confirmed - Group unknown
        # "^CERT[0-9]$": "CERT", # No info
        "^DCAF[0-9]$": "DCAF1",  # Confirmed - Group unknown
        "^TGM[0-9]$": "TGM",  # Confirmed + Group
        "^MAP3K21$": "MLK",  # !!!
        "^MAP3K20$": "STE11",  # !!!
        "^PHK[A-Z0-9]+$": "PHK",  # Questionable + Group
        "^FAM[0-9A-Z]+$": "GASK",  # Confirmed + Group
        "^SNF[0-9A-Z]+$": "SIK",  # Confirmed + Group
        "^GTF[0-9A-Z]+$": "TFIIF",  # Confirmed - Group unknown
        "^EPH[0-9A-Z]+$": "EPH",  # Confirmed + Group
        "^UL[0-9A-Z]+$": "HCMV",  # Confirmed + Group
        "^JMJD[0-9A-Z]+$": "JMJD6",  # Confirmed - Group unknown
        "^.*RET.*$": "RET",  # Questionable
        "^AURK.*$": "AUR",  # Confirmed + Group
        "^.*ABL.*$": "ABL",  # No info
        "HASPIN": "HASPIN",  # Confirmed - Group unknown
    }

    additional_group = {
        "CILK": "CMGC",
        "NDK": "ATYPICAL",
        "TGM": "ATYPICAL",
        "FAM": "ATYPICAL",
        "SIK": "CAMK",
        "TFIIF": "ATYPICAL",
        "HCMV": "TK",
        "PIK": "ATYPICAL",
        "PIPK": "ATYPICAL",
        "HASPIN": "OTHER",
    }

    checkpoints = []
    not_found = 1
    for i, r in combined_df[combined_df["Family"].isna()].iterrows():
        checkpoints.append(i)
        for a in additional:
            if bool(re.search(a, r["Kinase"])):
                combined_df.at[i, "Family"] = re.sub(f"{a}", f"{additional[a]}", r["Kinase"])
                combined_df.at[i, "Kinase"] = f"({r['Kinase']})"
                break
        else:
            assert not all(combined_df["Kinase"].isin(combined_df["Family"]))
            combined_df.at[i, "Family"] = f"{r['Kinase']}"
            combined_df.at[i, "Kinase"] = f"*{r['Kinase']}"
            logger.warning(f"No family found for {r['Kinase']}.")
            not_found += 1
    not_found = 1
    for i, r in combined_df[combined_df["Group"].isna()].iterrows():
        for ag in additional_group:
            if bool(re.search(ag, str(r["Family"]))):
                combined_df.at[i, "Group"] = additional_group[ag]
                combined_df.at[i, "Kinase"] = f"({r['Kinase']})"
                break
        else:
            if (
                len(
                    check := combined_df[
                        (combined_df["Family"] == combined_df.at[i, "Family"])
                        & (combined_df["Group"] != "<UNANNOTATED>")
                        & (combined_df["Group"].notna())
                    ]
                )
                != 0
            ):
                combined_df.at[i, "Group"] = check.iloc[0]["Group"]
            else:
                combined_df.at[i, "Group"] = "<UNANNOTATED>"
                logger.warning(f"No group found for {r['Kinase']}.")
                not_found += 1

    for i, r in combined_df[combined_df.isnull().any(axis=1)].iterrows():
        if ~pd.notna(r["Kinase"]):
            combined_df.at[i, "Kinase"] = f"[[{r['Uniprot']}]]"

    assert not any(pd.isnull(combined_df).values.ravel().tolist())
    # combined_df = combined_df.applymap(up)
    combined_df = combined_df.sort_values(
        kind="stable",
        key=lambda c: c.apply(
            lambda d: d.replace("(", "").replace(")", "").replace("*", "").replace("[[", "").replace("]]", "")
        ),
        by=["Kinase", "Family", "Group"],
    )
    combined_df = combined_df.applymap(up)

    held_out_df = combined_df[combined_df["Family"] == HELD_OUT_FAMILY]
    combined_df.drop(held_out_df.index, axis=0, inplace=True)

    combined_df.to_csv(fn := f"../kin_to_fam_to_grp_{len(combined_df)}.csv", index=False)
    held_out_df.to_csv(f"../kin_to_fam_to_grp_{len(held_out_df)}.csv", index=False)
    return fn


if __name__ == "__main__":
    get_kin_to_fam_to_grp("../../raw_data/kinase_seq_822.txt")
    print("(<kinase>) - Kinase with inferred family.")
    print("(*<kinase>) - Kinase with no family found. Setting family to name of kinase.")
