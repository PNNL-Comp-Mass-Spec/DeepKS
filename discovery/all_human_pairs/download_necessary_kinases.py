# %%
import pandas as pd, os, re, collections, asyncio, pathlib, warnings
from ..discovery_preparation import seq_request

async def main():
# %%
    os.chdir(pathlib.Path(__file__).parent.resolve())

    PSP_symbols = pd.read_csv("../data/raw_data/raw_data_22588.csv")
    PSP_symbols = PSP_symbols[PSP_symbols["organism"] == "HUMAN"]
    PSP_symbols = sorted((PSP_symbols["Kinase Sequence"] + "|" + PSP_symbols["uniprot_id"]).unique().tolist())
    PSP_exclusion_dict = { # True means keep, False means remove
        "BCKDK|O14874": True,
        "BCR/ABL FUSION|A9UF07": True,
        "BLVRA|P53004": False,
        "BRD4|O60885": True,
        "BRSK1|Q8TDC3-2": False,
        "CAMK2D|Q13557-8": False,
        "CDK11A|Q9UQ88-10": False,
        "CERT1|Q9Y5P4": False,
        "CSNK2B|P67870": True,
        "ENPP3|O14638": False,
        "GSK3B|P49841-2": False,
        "GTF2F1|P35269": True, # Ask
        "HSPA5|P11021": False,
        "JMJD6|Q6NYC1": False,
        "MAPK8|P45983-2": False,
        "MAPK9|P45984-2": False,
        "MARK3|P27448-3": False,
        "MKNK1|Q9BUB5-2": False,
        "NME1|P15531": True,
        "NME2|P22392": True,
        "PCK1|P35558": True, # Ask
        "PDK1|Q15118": True,
        "PDK2|Q15119": True,
        "PDK3|Q15120": True,
        "PDK4|Q16654": True,
        "PFKP|Q01813": False,
        "PGK1|P00558": False,
        "PHKA1|P46020": True,
        "PIK3C2A|O00443": False,
        "PIK3CB|P42338": True,
        "PIK3CD|O00329": False,
        "PIK3R1|P27986": False,
        "PKM|P14618": True,
        "PKM|P14618-2": False,
        "PRKAB1|Q9Y478": False,
        "PRKACA|P17612-2": False,
        "PRKAG2|Q9UGJ0": False,
        "PRKCB|P05771-2": False,
        "PRKG1|Q13976-2": False,
        "RET/PTC2|Q15300": True,
        "ROR1|Q01973": True,
        "RPS6KB1|P23443-2": False,
        "TGM2|P21980": False,
        "VRK2|Q86Y07-2": False,
        "VRK3|Q8IV63": False
    }
    print(f"{len(PSP_symbols)=}")
    PSP_filtered = [x for x in PSP_symbols if not re.search(r".*\|[0-9A-Z]+-[0-9]+", x)] # Remove All Remaining Isoforms
    print(f"{len(PSP_filtered)=}")
    PSP_filtered = [x for x in PSP_symbols if x not in PSP_exclusion_dict or PSP_exclusion_dict[x]]
    print(f"{len(PSP_filtered)=}")
    PSP_symbols = PSP_filtered

    # %%
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning, message="Workbook contains no default style")
        with open ("../images/Kinase Overlap/Uniprot_ST_Kinases.xlsx", "rb") as st:
            st_kinases = pd.read_excel(st)
        with open ("../images/Kinase Overlap/Uniprot_Y_Kinases.xlsx", "rb") as y:
            y_kinases = pd.read_excel(y)
    all_uniprot = pd.concat([st_kinases, y_kinases], ignore_index=True).sort_values(by = ["Entry Name", "Gene Names (primary)"])[['Entry', 'Entry Name', 'Gene Names (primary)']].rename(columns={'Entry': 'Uniprot ID', 'Gene Names (primary)': 'Gene Name'}).reset_index(drop=True)
    all_uniprot['Symbol'] = all_uniprot['Gene Name'] + "|" + all_uniprot['Uniprot ID']
    Uniprot_symbols = sorted(all_uniprot['Symbol'].unique().tolist())

    # %%
    all_symbols_set = set.union(set(Uniprot_symbols), set(PSP_symbols))
    all_symbols = sorted(list(all_symbols_set))

    # %%
    existing_symbol_to_seq = pd.read_csv("../data/raw_data/kinase_seq_826.csv")
    existing_symbol_to_seq["symbol"] = existing_symbol_to_seq["gene_name"] + "|" + existing_symbol_to_seq["kinase"]
    existing_symbol_to_seq_df = existing_symbol_to_seq.copy(deep=True)
    existing_symbol_to_seq = existing_symbol_to_seq.set_index("symbol").to_dict()["kinase_seq"]
    existing_symbol_to_seq = collections.OrderedDict(
        sorted({k: v for k, v in existing_symbol_to_seq.items() if k in all_symbols_set}.items())
    )

    # %%
    need_sequences = sorted([x for x in all_symbols_set if x not in existing_symbol_to_seq])
    assert all([isinstance(x, str) for x in need_sequences])

    # %%
    df_new = await seq_request(uniprot_ids=[str(x).split("|")[1] for x in need_sequences])


    # %%
    assert isinstance(df_new, pd.DataFrame)
    # Gene Name,Sequence,Uniprot ID,Name,Symbol
    new_raw_data = pd.DataFrame({'kinase': df_new['Uniprot ID'], 'kinase_seq': df_new['Sequence'], 'gene_name': df_new['Gene Name']})
    all_df = pd.concat([existing_symbol_to_seq_df[existing_symbol_to_seq_df['symbol'].isin(all_symbols)], new_raw_data], ignore_index=True).sort_values(by = ["gene_name", "kinase"]).reset_index(drop=True)

    # %%
    assert(len(all_df)) == len(all_symbols_set), f"{len(all_df)} != {len(all_symbols_set)}"

    # %%
    all_df.to_csv(f"../data/raw_data/kinase_seq_{len(all_df)}.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())