import requests, pandas as pd, gzip, io, re, os

def get_phospho(redownload = False, outfile = "PSP_script_download.xlsx"):
    url = "https://www.phosphosite.org/downloads/Kinase_Substrate_Dataset.gz"
    if not os.path.exists("Kinase_Substrate_Dataset.gz") and not redownload:
        try:
            r = requests.request("GET", url, headers={'Accept-Encoding': 'gzip, deflate, br', 'Accept': '*/*', 'Connection': 'keep-alive', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            if r.status_code == 200:
                with open("Kinase_Substrate_Dataset.gz", "wb") as f:
                    f.write(r.content)
            else:
                raise requests.HTTPError(str(r.status_code) + str(r.content))
        except Exception as e:
            print(e.__class__.__name__, e)
            exit(1)

    str_table = "\n".join(str(gzip.open("Kinase_Substrate_Dataset.gz", 'rb').read()).split("\\n")[3:]).replace("\\t", ",")
    table = pd.read_csv(io.StringIO(str_table))
    table = table[~table['GENE'].isnull() & ~table['KIN_ACC_ID'].isnull() & ~table['SITE_+/-7_AA'].isnull()] # Remove rows with crucial missing values
    table = table[~table['KIN_ACC_ID'].isin(['Q8UWG6', 'Q7M0H9', 'AAA58698', 'Q8WZ42'])] # Remove incorrect gene names/too long of sequences
    # EXCEPTIONS
    table = table.replace("NP_001100256", "D4A7D3").replace("NP_001178933", "D3Z8E0").replace("YP_068414", "Q05608").replace("NP_001178650", "F1LYL8").replace("NP_001100045", "B1WBT4") # Replace incorrect uniprot IDs
    get_official_names_for = table["KIN_ACC_ID"].unique()
    query = "+OR+".join([f"(accession:{upid.split('-')[0]})" for upid in get_official_names_for])
    new_url = f"https://rest.uniprot.org/uniprotkb/stream?format=json&query=({query})&fields=gene_primary"
    r = requests.get(new_url)
    if r.status_code == 200:
        gene_names = {x["primaryAccession"]: x["genes"][0]["geneName"]["value"] for x in r.json()['results']}
        for i, r in table.iterrows():
            if bool(re.match(".*-[0-9]+$", r['KIN_ACC_ID'])):
                gn = gene_names["-".join(r['KIN_ACC_ID'].split("-")[:-1])]
            else:
                gn = gene_names[r['KIN_ACC_ID']]
            table.at[i, "GENE"] = gn
    else:
        print(f"Error: {r.status_code}")
        if r.status_code == 400:
            print(r.json()['messages'])
            print("^^^The message above is likely due to an accession number not being found in the Uniprot database.")
            print("Please manually look up this invalid accession on Uniprot.org and add to `download_psp.py` under \"EXCEPTIONS\"")
        print("Exiting unsuccessfully.")
        exit(1)
    
    table = table.sort_values(by=['GENE', 'KIN_ORGANISM'])
    print("Info: Number of unique uniprot IDs in PSP: ", len(table["KIN_ACC_ID"].unique()))
    print("Status: Saving PSP to Excel file...")
    table.to_excel(outfile, index = False)

if __name__ == "__main__":
    get_phospho()