# %% ### IMPORTS ---
import pandas as pd, json, re, os, pathlib, plotly.graph_objects as go
from matplotlib import rcParams

rcParams["font.family"] = "P052-Roman"

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)

# %% ### DATA FORMATTING ---
def make_sunburst():
    kfg = pd.read_csv("../../data/preprocessing/kin_to_fam_to_grp_826.csv")
    kfg["Symbol"] = [re.sub(r"[\(\)\*]", "", x) for x in kfg["Kinase"] + "|" + kfg["Uniprot"]]

    kinase_to_family = kfg.set_index("Symbol").to_dict()["Family"]

    family_to_group = kfg.set_index("Family").to_dict()["Group"]

    new_df = pd.DataFrame({"Kinase": [], "Family": [], "Group": [], "MLSet": []})
    for fn, label in zip(
        [
            "../../data/preprocessing/tr_kins.json",
            "../../data/preprocessing/vl_kins.json",
            "../../data/preprocessing/te_kins.json",
        ],
        ["train", "val", "test"],
    ):
        with open(fn, "r") as f:
            set_of_kinases = [re.sub(r"[\(\)\*]", "", x) for x in json.load(f)]
            for kin in set_of_kinases:
                # kin = re.sub(r"[\(\)\*]", "", kin)
                new_df.loc[len(new_df)] = [None, None, None, None]  # type: ignore
                r = new_df.loc[len(new_df) - 1]
                r["Kinase"] = kin
                r["Family"] = kinase_to_family[kin] + "@F"
                r["Group"] = (
                    family_to_group[kinase_to_family[kin]] + "@G"
                    if kinase_to_family[kin] in family_to_group
                    else "<UNKNOWN>@G"
                )
                r["MLSet"] = label

    num_sites_df = pd.read_csv("../../data/raw_data/raw_data_22588.csv").rename({"lab": "Kinase"}, axis="columns")
    new_df = pd.merge(new_df, num_sites_df, how="left", on="Kinase").drop_duplicates(keep="first").reset_index(drop=True)
    num_sites_df["Symbol"] = num_sites_df["Kinase"] + "|" + num_sites_df["uniprot_id"]

    kin_to_num_sites = num_sites_df.set_index("Symbol").to_dict()["num_sites"]


    def get_sectors(df: pd.DataFrame):
        children = []
        parents = []
        for c, col in tuple(enumerate(df.columns))[1:]:
            col_iter = df[col]
            col_iter.index = pd.Index([x for x in range(len(df[col]))])
            for r, row in col_iter.items():
                assert isinstance(r, int)
                children.append(row)
                parents.append(df[df.columns[c - 1]].iloc[r])
        df_len = len(df)
        children += df[df.columns[0]].tolist()
        parents += ["" for _ in range(df_len)]
        return (
            pd.DataFrame(
                {
                    "labels": children,
                    "parents": parents,
                    "val": [0 if "@" in children[i] else 1 for i in range(len(children))],
                }
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

    cols: list[str] = new_df.columns.to_list()
    cols.remove("num_sites")
    sector_df: pd.DataFrame = new_df[cols].copy()[["Group", "MLSet", "Family", "Kinase"]]

    sector_df.sort_values(["Group", "MLSet", "Family", "Kinase"])

    def make_unq(row: pd.Series):
        row["MLSet"] += "@@" + row["Group"]
        return row

    sector_df = sector_df.apply(make_unq, axis=1)
    sectors = get_sectors(sector_df)
    sectors["val"] = [0 if "@" in x["labels"] else kin_to_num_sites[x["labels"]] for _, x in sectors.iterrows()]

    # %% ### MAIN SUNBURST PLOT ---
    fig = go.Figure(
        go.Sunburst(
            ids=sectors["labels"],
            labels=[x.split("@@")[0].split("@")[0] for x in sectors["labels"]],
            parents=sectors["parents"],
            values=sectors["val"],
            insidetextfont=go.sunburst.Insidetextfont(family="P052-Roman", size=24),
            insidetextorientation="radial",
        )
    )

    fig.update_layout(autosize=False, width=1000, height=1000, margin=dict(t=0, l=0, r=0, b=0))

    # fig.show()
    with open("sunburst.pdf", "wb") as f:
        f.write(fig.to_image(format="pdf", height=1000, width=1000))


    # %% ### EXPLAINER PLOT ---
    AA = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
    assert len(AA) == 20
    explainer = pd.read_excel("./Onion Explainer.xlsx", sheet_name="Main Detail (2)")
    explainer.index = pd.Index([x for x in range(len(explainer))])
    fake_kins = iter([f"Seq-{x}" for x in range(sum(explainer["val"]))])
    for i, r in explainer.iterrows():
        assert isinstance(i, int)
        if i > 29:
            next_kins = [fake_kins.__next__() for _ in range(r["val"])]
            next_kins_iterator = iter(next_kins)
            for j in range(len(explainer), len(explainer) + r["val"]):
                fake_kin = next_kins_iterator.__next__()
                explainer.loc[j] = [fake_kin, "Site", r["id"], 1]  # type: ignore
            explainer.at[i, "val"] = 0


    sb = go.Sunburst(
        ids=explainer["id"],
        labels=explainer["labels"],
        parents=explainer["parents"],
        values=explainer["val"],
        insidetextfont=go.sunburst.Insidetextfont(family="P052-Roman", size=20),
        insidetextorientation="radial",
    )

    fig = go.Figure(sb)
    fig.update_layout(autosize=False, width=1000, height=1000, margin=dict(t=0, l=0, r=0, b=0))

    # fig.show()
    with open("sunburst explainer.pdf", "wb") as f:
        f.write(fig.to_image(format="pdf", height=1000, width=1000))

    # %% ### COMPANION PLOT ---
    companion = pd.read_excel("./Onion Explainer.xlsx", sheet_name="Companion")
    fig = go.Figure(
        go.Sunburst(
            ids=companion["id"],
            labels=companion["labels"],
            parents=companion["parents"],
            values=companion["val"],
            insidetextfont=go.sunburst.Insidetextfont(family="P052-Roman", size=20),
            insidetextorientation="radial",
        )
    )

    fig.update_layout(autosize=False, width=1000, height=1000, margin=dict(t=0, l=0, r=0, b=0))

    # fig.show()
    with open("sunburst explainer simple.pdf", "wb") as f:
        f.write(fig.to_image(format="pdf", height=1000, width=1000))

# %% ### MAIN ---
if __name__ == "__main__":
    make_sunburst()