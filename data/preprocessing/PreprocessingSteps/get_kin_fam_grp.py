import re, io, pandas as pd, requests as req

HELD_OUT_FAMILY = "SGK"

def get_kin_to_fam_to_grp(relevant_kinases):
    up = lambda x: x.upper() if isinstance(x, str) else x
    url = "http://www.kinhub.org/kinases.html"
    r = req.request("GET", url)
    if r.status_code == 200:
        table = str(r.content)
    else: 
        raise req.HTTPError(str(r.status_code) + str(r.content))
    table_io = io.StringIO(table)
    table_pd = pd.read_html(table_io)[0]
    table_pd.rename(columns = {'HGNC Name\\n \\n': 'Kinase'}, inplace = True)
    table_pd = table_pd[~table_pd['Manning Name\\n \\n'].str.contains('Domain2')][['Kinase', 'Family', 'Group']] # .to_csv('../preprocessing/kin_to_fam_to_grp.csv', index = False)
    table_pd = table_pd[~table_pd.isnull().any(axis=1)]

    table_pd = table_pd.applymap(up)
    relevant_kinases = pd.read_csv(relevant_kinases, sep = '\t')[['kinase', 'gene_name']].rename({'kinase': 'Uniprot', 'gene_name': 'Kinase'}, axis = 1).applymap(up)

    combined_df = pd.merge(relevant_kinases, table_pd, how = "left", on = 'Kinase')
    combined_df = combined_df[combined_df['Kinase'].isin(relevant_kinases['Kinase'])]

    additional = {
        "^PIK[0-9A-Z]+$": "PIK", 
        "^CDK[0-9A-Z]+$": "CDK", 
        "^NME[0-9]$": "NDP", 
        "^PAK[0-9]": "STE20", 
        "^GRK[0-9]": "GRK", 
        "^PRKA[A-Z][0-9]$": "CAMKL",
        "^ENPP[0-9]$": "ENPP",
        "^CSNK2.*$": "CK2",
        "^CILK[0-9]$": "CILK",
        "^PGK[0-9]$": "PGK",
        "^PCK[0-9]$": "PCK",
        "^ALDO.*$": "ALDO",
        "^CERT[0-9]$": "CERT",
        "^DCAF[0-9]$": "DCAF",
        "^TGM[0-9]$": "TGM",
        "^MAP3K[0-9]+$": "STE11",
        "^PHK[A-Z0-9]+$": "PHK",
        "^FAM[0-9A-Z]+$": "FAM",
        "^SNF[0-9A-Z]+$": "SNF",
        "^GTF[0-9A-Z]+$": "GTF",
        "^EPH[0-9A-Z]+$": "EPH",
        "^UL[0-9A-Z]+$": "UL",
        "^JMJD[0-9A-Z]+$": "JMJD",
        "^.*RET.*$": "RET",
        "^AURK.*$": "AUR",
        "^.*ABL.*$": "ABL",
        }

    checkpoints = []
    not_found = 1
    for i, r in combined_df[~combined_df['Family'].notna()].iterrows():
        checkpoints.append(i)
        for a in additional:
            if bool(re.match(a, r['Kinase'])):
                combined_df.at[i, 'Family'] = re.sub(f"{a}", f"{additional[a]}", r['Kinase'])
                combined_df.at[i, 'Kinase'] = f"({r['Kinase']})"
                break
        else:
            assert not all(combined_df['Kinase'].isin(combined_df['Family']))
            combined_df.at[i, 'Family'] = f"{r['Kinase']}"
            combined_df.at[i, 'Kinase'] = f"*{r['Kinase']}"
            print(f"{not_found}. No family found for {r['Kinase']}.")
            not_found += 1
        if len(check := combined_df[(combined_df['Family'] == combined_df.at[i, 'Family']) & (combined_df['Group'] != "<UNANNOTATED>") & (combined_df['Group'].notna())]) != 0:
            combined_df.at[i, 'Group'] = check.iloc[0]['Group']
        else:
            combined_df.at[i, "Group"] = "<UNANNOTATED>"
        
    for i, r in combined_df[combined_df.isnull().any(axis=1)].iterrows():
        if ~pd.notna(r['Kinase']):
            combined_df.at[i, 'Kinase'] = f"[[{r['Uniprot']}]]"

    assert(not any(pd.isnull(combined_df).values.ravel().tolist()))
    # combined_df = combined_df.applymap(up)
    combined_df = combined_df.sort_values(key= lambda c: c.apply(lambda d: d.replace("(", "").replace(")", "").replace("*", "").replace("[[", "").replace("]]", "")), by = ['Kinase', 'Family', 'Group'])
    combined_df = combined_df.applymap(up)

    held_out_df = combined_df[combined_df['Family'] == HELD_OUT_FAMILY]
    combined_df.drop(held_out_df.index, axis = 0, inplace=True)


    combined_df.to_csv(f"../kin_to_fam_to_grp_{len(combined_df)}.csv", index = False)
    held_out_df.to_csv(f"../kin_to_fam_to_grp_{len(held_out_df)}.csv", index = False)

if __name__ == "__main__":
    get_kin_to_fam_to_grp("../../raw_data/kinase_seq_822.txt")