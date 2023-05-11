"""Downloads the most recent version of the PhosphoSitePlus database and saves it to an Excel file."""

import requests, pandas as pd, gzip, io, re, os, json, pathlib

from ....config.logging import get_logger

logger = get_logger()
"""The logger for this module."""
if __name__ == "__main__": # pragma: no cover
    logger.status("Loading Modules")

def get_phospho(redownload=False, outfile=(outfile := "PSP_script_download_debug.xlsx")):
    """Downloads the most recent version of the PhosphoSitePlus database and saves it to an Excel file.

    Parameters
    ----------
    redownload : bool, optional
        Whether or not to redownload the excel file, by default False
    outfile : tuple, optional
        The save location for the PSP database, by default (outfile := "PSP_script_download_debug.xlsx")

    Raises
    ------
    requests.HTTPError
        If the download fails
    """
    url = "https://www.phosphosite.org/downloads/Kinase_Substrate_Dataset.gz"
    if not os.path.exists("Kinase_Substrate_Dataset.gz") and not redownload:
        try:
            r = requests.request(
                "GET",
                url,
                headers={
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept": "*/*",
                    "Connection": "keep-alive",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                        " Chrome/91.0.4472.124 Safari/537.36"
                    ),
                },
                timeout=10,
            )
            if r.status_code == 200:
                with open("Kinase_Substrate_Dataset.gz", "wb") as f:
                    f.write(r.content)
            else:
                raise requests.HTTPError(str(r.status_code) + str(r.content))
        except Exception as e:
            print(e.__class__.__name__, e)
            exit(1)

    str_table = "\n".join(str(gzip.open("Kinase_Substrate_Dataset.gz", "rb").read()).split("\\n")[3:]).replace(
        "\\t", ","
    )
    table = pd.read_csv(io.StringIO(str_table))
    table = table[
        ~table["GENE"].isnull() & ~table["KIN_ACC_ID"].isnull() & ~table["SITE_+/-7_AA"].isnull()
    ]  # Remove rows with crucial missing values
    with open(f"{pathlib.Path(__file__).parent.resolve()}/psp_exceptions.json", "r") as f:
        excs = json.load(f)
        table = table[
            ~table["KIN_ACC_ID"].isin(excs["Accession Removals"])
        ]  # Remove incorrect gene names/too long of sequences # TODO - do this programatically

        reps = excs["Accession Replacements"]
        for rep in reps:
            table = table.replace(rep, reps[rep])  # Replace incorrect uniprot IDs

    get_official_names_for = table["KIN_ACC_ID"].unique()
    query = "+OR+".join([f"(accession:{upid.split('-')[0]})" for upid in get_official_names_for])
    new_url = f"https://rest.uniprot.org/uniprotkb/stream?format=json&query=({query})&fields=gene_primary"
    MAX_TRIES = 5
    tries = 0
    done = False
    while not done and tries <= MAX_TRIES:
        tries += 1
        r = requests.get(new_url, timeout=10)
        if r.status_code == 200:
            done = True
            gene_names = {x["primaryAccession"]: x["genes"][0]["geneName"]["value"] for x in r.json()["results"]}
            for i, r in table.iterrows():
                if bool(re.search(".*-[0-9]+$", r["KIN_ACC_ID"])):
                    gn = gene_names["-".join(r["KIN_ACC_ID"].split("-")[:-1])]
                else:
                    gn = gene_names[r["KIN_ACC_ID"]]
                table.at[i, "GENE"] = gn
        else:
            print(f"Error: {r.status_code}")
            if r.status_code == 400:
                print(r.json()["messages"])
                print(
                    "^^^The message above is likely due to a PSP entry not being compatible with the Uniprot"
                    " Database.\nPlease determine which entry is causing the problem and add it appropriately to"
                    f" {pathlib.Path(__file__).parent.resolve()}/psp_exceptions.json."
                )
            print("Exiting unsuccessfully.")
            exit(1)

    table = table.sort_values(by=["GENE", "KIN_ORGANISM"])
    logger.info(f"Info: Number of unique uniprot IDs in PSP: {len(table['KIN_ACC_ID'].unique())}")
    logger.status("Saving PSP to Excel file...")
    table.to_excel(outfile, index=False)


if __name__ == "__main__": # pragma: no cover
    get_phospho()
