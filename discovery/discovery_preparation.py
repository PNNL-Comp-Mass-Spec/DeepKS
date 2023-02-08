# %%
# Imports and formatting

import pandas as pd, numpy as np, re, os, sys, collections, requests, asyncio, aiohttp, itertools, tqdm, time, pathlib, json
from typing import List, Coroutine
from pprint import pprint

np.set_printoptions(precision=3, edgeitems=10, linewidth=180)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 180)
pd.set_option("display.min_rows", 50)
UNIPROT_REQUEST_SIZE = 400

DO_REQUEST = not os.path.exists("sequences_table.csv")

# %%
# Get the phosphosite data into appropriate format
all_phos_sites_to_loc = collections.defaultdict(list[str])
base_series: list[str] = pd.read_csv("phosphosites_base.txt", sep="\t")["feature_names"].tolist()

for x in base_series:
    all_phos_sites_to_loc[x.split("-")[0]] += sorted(x.split("-")[1].split(","))

for k in all_phos_sites_to_loc.keys():
    all_phos_sites_to_loc[k] = sorted(list(set(all_phos_sites_to_loc[k])))

all_phos_sites_to_loc = collections.OrderedDict(sorted(all_phos_sites_to_loc.items(), key=lambda x: x[0]))

if DO_REQUEST:
    # %% [markdown]
    # ### Only Need to be run once.

    # %%
    # Make API requests to Uniprot to get the sequence of each protein

    organism = "9606"

    query_part_1 = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&uncompressed=true&query=reviewed:true+AND+organism_id:{organism}+AND+"
    gene_queries = [
        query_part_1
        + "("
        + "+OR+".join([f"gene:{x}" for x in list(all_phos_sites_to_loc.keys())[i : i + UNIPROT_REQUEST_SIZE]])
        + ")"
        for i in range(0, len(all_phos_sites_to_loc), UNIPROT_REQUEST_SIZE)
    ]

    # %%
    from async_timeout import timeout

    MAX_TRIES = 4

    async def get_url(url, session: aiohttp.ClientSession):
        done = False
        tries = 0
        while not done and tries < MAX_TRIES:
            try:
                print("About to request.")
                async with session.get(url, timeout=10) as r:
                    if r.status != 200:
                        print("---- RESPONSE TEXT -------")
                        print(await r.text())
                        print("--------------------------")
                        raise requests.HTTPError(f"Request failed with status code {r.status}. Response text above.")
                    return await r.text()

            except Exception as e:
                print(str(e.__class__.__name__) + ":", e)
                tries += 1
                print(f"Retrying in {tries*10} seconds.")
        raise RuntimeError(f"Failed to get URL after {MAX_TRIES} tries.")

    # %%
    async def get_fasta_pages() -> list[str]:
        MAX_CONNECTIONS = 1
        SLEEP = 1

        fasta_pages: list[str] = []
        async with aiohttp.ClientSession() as session:
            for i in tqdm.tqdm(range(0, len(gene_queries), MAX_CONNECTIONS), colour="cyan"):
                if i != 0:
                    time.sleep(SLEEP)
                fasta_page: list[str] = await asyncio.gather(
                    *[get_url(url, session) for url in gene_queries[i : i + MAX_CONNECTIONS]]
                )
                fasta_pages += fasta_page
        return fasta_pages

    fasta_pages = asyncio.run(get_fasta_pages())

    # %%
    all_fasta_string = "".join(fasta_pages)

    # %%
    names = re.findall(r"GN=([^\s]+)", all_fasta_string)
    sequences = [x.replace("\n", "") for x in re.findall(r">.*\n([^>]+)", all_fasta_string)]
    ids = re.findall(r">.*?\|(.*?)\|", all_fasta_string)
    names_long = re.findall(r">.*?\|.*?\|(.*?) OS=", all_fasta_string)
    assert len(names) == len(sequences) == len(ids) == len(names), "Lengths of fasta information are not equal."

    sequences_table = pd.DataFrame({"Gene Name": names, "Sequence": sequences, "Uniprot ID": ids, "Name": names_long})
    sequences_table["Symbol"] = sequences_table["Gene Name"] + "|" + sequences_table["Uniprot ID"]
    sequences_table.to_csv("sequences_table.csv", index=False)

# %% [markdown]
# ### Continue from here.

# %%
sequences_table = pd.read_csv("sequences_table.csv")

# %%
gene_name_to_uniprot = collections.OrderedDict(
    sorted(sequences_table.set_index("Gene Name").to_dict()["Uniprot ID"].items())
)
gene_name_to_sequence = collections.OrderedDict(
    sorted(sequences_table.set_index("Gene Name").to_dict()["Sequence"].items())
)

# %%
count = 0
for g in all_phos_sites_to_loc.keys():
    if g not in gene_name_to_sequence:
        count += 1
        print(f"{count}. WARNING: Gene {g} not found in reviewed Uniprot database.")

# %%
FLANKING_OFFSET = 7
warn_num = 0
symbol_to_location = collections.defaultdict(list[str])
symbol_to_flanking_sites = collections.defaultdict(list[str])
for gene_name in all_phos_sites_to_loc.keys():
    if gene_name in gene_name_to_sequence:
        sequence = gene_name_to_sequence[gene_name]
        for site in all_phos_sites_to_loc[gene_name]:
            middle = int(site[1:]) if site[1:].isnumeric() else -1
            if middle == -1 or middle >= len(sequence):
                warn_num += 1
                print(
                    f"{warn_num}.",
                    "Warning: Site",
                    site,
                    "is out of bounds or invalid for gene",
                    gene_name,
                    "with sequence length",
                    len(sequence),
                    "So skipping.",
                )
                continue
            symbol_to_location[gene_name + "|" + gene_name_to_uniprot[gene_name]].append(
                sequence[middle - 1] + str(middle - 1)
            )
            if sequence[middle - 1] not in ["S", "T", "Y"]:
                warn_num += 1
                print(
                    f"{warn_num}. Warning: Site {site} for gene {gene_name} is not a phosphorylation site (central AA ="
                    f" {sequence[middle - 1]}). Skipping."
                )
            if middle - FLANKING_OFFSET - 1 < 0:
                l = 0
                lpart = "X" * abs(middle - FLANKING_OFFSET - 1)
            else:
                l = middle - FLANKING_OFFSET - 1
                lpart = ""
            if middle + FLANKING_OFFSET > len(sequence):
                r = len(sequence)
                rpart = "X" * (FLANKING_OFFSET - (len(sequence) - middle))
            else:
                r = middle + FLANKING_OFFSET
                rpart = ""
            flank_seq = lpart + sequence[l:r] + rpart
            assert len(flank_seq) == 2 * FLANKING_OFFSET + 1, (
                f"Flanking sequence {flank_seq} is not of length {2*FLANKING_OFFSET + 1} for site {site} for gene"
                f" {gene_name} with sequence length {len(sequence)}."
            )
            symbol_to_flanking_sites[gene_name + "|" + gene_name_to_uniprot[gene_name]].append(
                lpart + sequence[l:r] + rpart
            )

symbol_to_flanking_sites = collections.OrderedDict(sorted(symbol_to_flanking_sites.items(), key=lambda x: x[0]))
symbol_to_location = collections.OrderedDict(sorted(symbol_to_location.items(), key=lambda x: x[0]))
site_to_site_id = collections.OrderedDict(
    {s: i for i, s in sorted(enumerate(list(itertools.chain(*list(symbol_to_flanking_sites.values())))))}.items()
)

# %%
where_am_i = os.path.abspath("")
os.chdir(where_am_i)

# %%
kinase_symbols = pd.read_csv("../data/raw_data/raw_data_22588.csv")
kinase_symbols = kinase_symbols[kinase_symbols["organism"] == "HUMAN"]
relevant_kinase_symbols: list[str] = sorted(
    (kinase_symbols["lab"] + "|" + kinase_symbols["uniprot_id"]).unique().tolist()
)

# %%
ks = pd.read_csv("../data/raw_data/kinase_seq_826.csv")
kinase_symbol_to_kinase_sequence = collections.OrderedDict(
    sorted({ksymb: kseq for ksymb, kseq in zip(ks["gene_name"] + "|" + ks["kinase"], ks["kinase_seq"])}.items())
)
kinase_list = [kinase_symbol_to_kinase_sequence[x] for x in relevant_kinase_symbols]
site_list = list(site_to_site_id.keys())

SMALL_KIN = 20  # len(kinase_list)
SMALL_SITE = 50  # len(site_list)
kinase_symbol_list = relevant_kinase_symbols[:SMALL_KIN]
site_symbol_list = list(itertools.chain(*[[x] * len(symbol_to_location[x]) for x in list(symbol_to_location.keys())]))[
    :SMALL_SITE
]

kinase_list = kinase_list[:SMALL_KIN]
site_list = site_list[:SMALL_SITE]

with open(f"site_list_{len(site_list)}.txt", "w") as f, open(f"kinase_list_{len(kinase_list)}.txt", "w") as g:
    f.write("\n".join(site_list))
    g.write("\n".join(kinase_list))


# site_X_kinase = [(x[0], kinase_symbol_to_kinase_sequence[x[1]]) for x in site_X_kinase_symbol]
# site_list = [x[0] for x in site_X_kinase]
# kinase_list = [x[1] for x in site_X_kinase]

# %%

# kinase_gene_names = set([x.split("|")[0] for x in kinase_symbol_list])
# kinase_uniprot_ids = set([x.split("|")[1] for x in kinase_symbol_list])
# site_gene_names = set([x.split("|")[0] for x in site_symbol_list])
# site_uniprot_ids = set([x.split("|")[1] for x in site_symbol_list])

kinase_symbol_list = set(kinase_symbol_list)
site_symbol_list = set(site_symbol_list)

site_to_info = {}

for site_symbol in site_symbol_list:
    for i, flank_seq in enumerate(symbol_to_flanking_sites[site_symbol]):
        site_to_info[flank_seq] = {
            "Uniprot Accession ID": site_symbol.split("|")[1],
            "Gene Name": site_symbol.split("|")[0],
            "Location": symbol_to_location[site_symbol][i],
        }

kinase_to_info = {
    kinase_symbol_to_kinase_sequence[kinase_symbol]: {
        "Uniprot Accession ID": kinase_symbol.split("|")[1],
        "Gene Name": kinase_symbol.split("|")[0],
    }
    for kinase_symbol in kinase_symbol_list
}

with open(f"compact_kinase_info_{len(kinase_to_info)}.json", "w") as kf, open(
    f"compact_site_info_{len(site_to_info)}.json", "w"
) as sf:
    json.dump(kinase_to_info, kf, indent=3)
    json.dump(site_to_info, sf, indent=3)


# kinase_gene_names = list(
#     itertools.chain(
#         *[[x.split("|")[0]] * SMALL_SITE for x in list(kinase_symbol_to_kinase_sequence.keys())[:SMALL_KIN]]
#     )
# )
# kinase_uniprot_ids = list(
#     itertools.chain(
#         *[[x.split("|")[1]] * SMALL_SITE for x in list(kinase_symbol_to_kinase_sequence.keys())[:SMALL_KIN]]
#     )
# )

# site_locations = (
#     list(
#         itertools.chain(
#             *[
#                 symbol_to_location[x]
#                 for x in sorted(list(set(["|".join(x) for x in zip(site_gene_names, site_uniprot_ids)])))
#             ]
#         )
#     )[:SMALL_SITE]
#     * SMALL_KIN
# )
# site_gene_names = (
#     list(itertools.chain(*[[x.split("|")[0]] * len(symbol_to_location[x]) for x in list(symbol_to_location.keys())]))[
#         :SMALL_SITE
#     ]
#     * SMALL_KIN
# )
# site_uniprot_ids = (
#     list(itertools.chain(*[[x.split("|")[1]] * len(symbol_to_location[x]) for x in list(symbol_to_location.keys())]))[
#         :SMALL_SITE
#     ]
#     * SMALL_KIN
# )

# # %%
# assert len(site_gene_names) == len(site_uniprot_ids) == SMALL_KIN * SMALL_SITE, (
#     len(site_gene_names),
#     len(site_uniprot_ids),
#     SMALL_KIN * SMALL_SITE,
# )
# assert len(site_uniprot_ids) == len(site_locations) == SMALL_KIN * SMALL_SITE, (
#     len(site_uniprot_ids),
#     len(site_locations),
#     SMALL_KIN * SMALL_SITE,
# )
# assert len(site_locations) == len(kinase_gene_names) == SMALL_KIN * SMALL_SITE, (
#     len(site_locations),
#     len(kinase_gene_names),
# )
# assert len(kinase_gene_names) == len(kinase_uniprot_ids) == SMALL_KIN * SMALL_SITE, (
#     len(kinase_gene_names),
#     len(kinase_uniprot_ids),
#     SMALL_KIN * SMALL_SITE,
# )

# # %%
# kin_info_df = pd.DataFrame({"gene_name": kinase_gene_names, "gene_uniprot_id": kinase_uniprot_ids})
# site_info_df = pd.DataFrame(
#     {"gene_name": site_gene_names, "gene_uniprot_id": site_uniprot_ids, "gene_site": site_locations}
# )

# # %%
# kin_info_df.to_csv(f"kinase_info_{len(kinase_list)*len(site_list)}.csv", index=False)
# site_info_df.to_csv(f"site_info_{len(kinase_list)*len(site_list)}.csv", index=False)
