# ### Setup
from matplotlib.figure import Figure
import pandas as pd, tqdm, json
from transformers import T5Tokenizer, T5EncoderModel
import torch, re, random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "P052"
# %config InlineBackend.figure_formats = ['svg']
from sklearn import decomposition
import numpy as np

np.set_printoptions(suppress=True)


class ionoff:
    def __enter__(self):
        plt.ioff()

    def __exit__(self, type, value, traceback):
        plt.ion()
        plt.show(block=True)
        plt.close(plt.gcf())


# ### Embeddings
def get_embeddings(sequence_examples, model, device, tokenizer, chunk_size=None):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    assert tokenizer is not None
    # tokenize sequences and pad up to the longest sequence in the batch
    if chunk_size is None:
        if "cpu" in str(device):
            chunk_size = 1
        else:
            chunk_size = 18
    # generate embeddings
    with torch.no_grad():
        res = []
        ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
        for i in tqdm.tqdm(range(0, len(sequence_examples), chunk_size)):
            rng = slice(i, i + chunk_size)
            input_ids = torch.tensor(ids["input_ids"][rng]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"][rng]).to(device)
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

            emb_per_proteins = [embedding_repr.last_hidden_state[j][:-1].mean(dim=0) for j in range(len(input_ids))]

            embedded = [[float(ee) for ee in e.cpu().numpy().tolist()] for e in emb_per_proteins]
            res += embedded
    return res


def do_normalize(data: list[list[int | float]], mat=1, mit=-1):
    ma, mi = max([max(d) for d in data]), min([min(d) for d in data])
    ma, mi = float(ma), float(mi)
    return [[((mat - mit) * (x - mi) / (ma - mi)) + mit for x in d] for d in data]


# ### PCA


def get_pca(embeddings: np.ndarray, annotations: list[str], idx_to_color={}) -> Figure:
    with ionoff():
        assert len(annotations) == len(embeddings), (
            "The number of embeddings (i.e., sequences) equal the number of annotations (i.e., labels) for said"
            " embeddings."
        )
        pca = decomposition.PCA()
        X_trans = pca.fit_transform(embeddings)
        fig, ax = plt.subplots()
        ax.scatter(X_trans[:, 0], X_trans[:, 1], c=[idx_to_color.get(i, "blue") for i in range(len(X_trans))])
        for i in range(len(annotations)):
            plt.annotate(
                text="".join([se for se in annotations[i] if se != " "]),
                xy=(X_trans[i, 0], X_trans[i, 1]),
                fontfamily="Fira Code",
                fontweight=500,
                fontsize=8,
            )

        ax.set_title("PCA of ProtT5")
        ax.set_xlabel(f"First Component ({pca.explained_variance_ratio_[0]*100:2.2f}%)")
        ax.set_ylabel(f"Second Component ({pca.explained_variance_ratio_[1]*100:2.2f}%)")
    return fig


def get_scree(embeddings: np.ndarray) -> Figure:
    with ionoff():
        pca = decomposition.PCA()
        pca.fit(embeddings)
        fig, ax = plt.subplots()
        ax.plot(
            range(1, len(pca.explained_variance_ratio_) + 1),
            100 * pca.explained_variance_ratio_,
            "-bo",
            label="Variance Explained",
        )

        ax.plot(
            range(1, len(pca.explained_variance_ratio_) + 1),
            [sum(100 * pca.explained_variance_ratio_[:i]) for i in range(len(pca.explained_variance_ratio_))],
            "-go",
            label="Cum. Variance Explained",
        )

        ax.set_title("Scree Plot for ProtT5 PCA")
        ax.set_ylabel("% Variance Explained")
        ax.set_xlabel("Number of Principle Components")
        ax.set_xticks(list(range(len(pca.explained_variance_ratio_) + 1)))
        ax.set_yticks(list(range(0, 101, 10)), list(range(0, 101, 10)))
        ax.set_xlim(0.5, len(pca.explained_variance_ratio_) + 0.5)
        ax.set_ylim(-5, 105)
        _ = ax.legend()
        return fig


def main(sequence_examples=["PEPTIDE", "SEQWENCES"], pickle_loc=f"./embeddings"):
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model: torch.nn.Module = T5EncoderModel.from_pretrained(transformer_link)  # type: ignore
    model = model.to(device)
    if str(device) == "cpu":
        model = model.to(torch.float32)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    embeddings = get_embeddings(sequence_examples, model, device, tokenizer)
    # X_embeddings = np.array(embeddings)
    # get_scree(X_embeddings)
    # get_pca(X_embeddings, [f"{str(x)[:10]}..." for x in X_embeddings])


def kinase_pca(kinase_filename: str, sequence_col: str, anno_col: str, pickle_loc=f"./embeddings.json"):
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model: torch.nn.Module = T5EncoderModel.from_pretrained(transformer_link)  # type: ignore
    model = model.to(device)
    if str(device) == "cpu":
        model = model.to(torch.float32)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    df = pd.read_csv(kinase_filename)
    keep_indices = [int(i) for i, r in (~df[sequence_col].duplicated(keep="first")).items() if r]
    sequence_examples = df[sequence_col].loc[keep_indices].tolist()
    embeddings = get_embeddings(sequence_examples, model, device, tokenizer)
    with open(pickle_loc, "w") as f:
        json.dump(dict(zip(df[anno_col].loc[keep_indices], embeddings)), f, indent=3)
    # X_embeddings = np.array(embeddings)
    # get_scree(X_embeddings)
    # get_pca(X_embeddings, df[keep_indices, anno_col].tolist())


if __name__ == "__main__":
    kinase_pca(
        "/home/dockeruser/DeepKS2/DeepKS/data/raw_data_31834_formatted_65_26610.csv",
        "Kinase Sequence",
        "Gene Name of Provided Kin Seq",
    )
