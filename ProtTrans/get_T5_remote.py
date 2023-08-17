from __future__ import annotations
import json, re, os, numpy as np, torch, sys
from copy import deepcopy
from ..config.join_first import join_first
from ..config.logging import get_logger
from ..tools.custom_tqdm import CustomTqdm
from numpy.typing import NDArray as numpyArray

logger = get_logger()
"""Logger for this module."""

AAs = set(list("ACDEFGHIKLMNPQRSTVWXY"))


class GetT5:
    T5_CACHE_PATH = join_first("T5_cache.json", 0, __file__)
    t5_instance = None

    def __init__(self):
        try:
            with open(self.T5_CACHE_PATH, "r") as cache_fp:
                self.current = json.load(cache_fp)
        except FileNotFoundError:
            logger.uerror(
                f"File {self.T5_CACHE_PATH} not found.\nThis may mean there exists no T5 cache file.\nTo avoid"
                " accidental overwrites, please create it manually by running\n\nprintf '{}' >"
                f" {self.T5_CACHE_PATH}\n\nin the console."
            )
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.uerror(
                f"The json cache file {self.T5_CACHE_PATH} is corrupt. Consider running \n\nprintf '{{}}' >"
                f" {self.T5_CACHE_PATH}\n\nto reset it.\n\nHOWEVER, THE DATA IT CONTAINS WILL BE LOST."
            )
        except Exception as e:
            logger.error(e)
        self.new_embs = {}

    def get_t5(self, seqs: list[str] | numpyArray[np.str_], device: torch.device) -> torch.FloatTensor:
        for seq in seqs:
            assert isinstance(seq, str)
            assert all(s in AAs for s in seq), f"Found illegal AA character in {seq}."
        need_from_cache: list[int] = []
        need_from_model: list[int] = []
        for s, seq in enumerate(seqs):
            assert isinstance(seq, str)
            if seq in self.current:
                need_from_cache.append(s)
            else:
                need_from_model.append(s)

        res_cache = self._get_from_cache([seqs[s] for s in need_from_cache])
        if need_from_model:
            res_model = self._get_from_model([seqs[s] for s in need_from_model], device)
        else:
            res_model = []
        final_res = []
        need_from_cache_set = set(need_from_cache)
        need_from_model_set = set(need_from_model)
        for i in range(len(seqs)):
            if i in need_from_cache_set:
                final_res.append(deepcopy(res_cache[0]))
                del res_cache[0]
            elif i in need_from_model_set:
                final_res.append(deepcopy(res_model[0]))
                del res_model[0]
            else:
                raise AssertionError(
                    "Should never see this message. If one does, it means that an index into the list of sequences to"
                    " T5-embed was neither in `need_from_cache_set` nor `need_from_model_set`."
                )
        self._save_back()
        return torch.FloatTensor(final_res)

    def _save_back(self, save_back_path=T5_CACHE_PATH):
        with open(save_back_path.replace(".json", ".backup.json"), "w") as tempfp:
            json.dump(self.current, tempfp, indent=3)
        self.current = self.current | self.new_embs
        with open(save_back_path, "w") as fp:
            json.dump(self.current, fp, indent=3)
        os.unlink(tempfp.name)

    def _get_from_cache(self, seqs: list[str] | numpyArray[np.str_]) -> list[list[float]]:
        res = []
        for seq in seqs:
            assert (
                seq in self.current
            ), f"For some (strange) reason, the sequence {seq}... was not found in `self.current`."
            embedding = self.current[seq]
            assert isinstance(embedding, list)
            assert all(isinstance(e, float) for e in embedding)
            res.append(embedding)
        return res

    def _get_from_model(self, seqs: list[str] | numpyArray[np.str_], device: torch.device) -> list[list[float]]:
        if self.t5_instance is None:
            self.t5_instance = self.T5Instance(device, self)
        self.t5_instance.device = device
        ret = self.t5_instance.get_embeddings(seqs)
        return ret

    class T5Instance:
        def __init__(self, device: torch.device, outer: GetT5):
            self.device = device
            self.outer = outer

        def get_embeddings(self, seqs: list[str] | numpyArray[np.str_]) -> list[list[float]]:
            from transformers import T5Tokenizer, T5EncoderModel

            logger.status(f"Getting embeddings from the T5 model from scratch for {len(seqs)} sequence(s).")
            transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
            logger.status("Loading: {}".format(transformer_link))
            model: torch.nn.Module = T5EncoderModel.from_pretrained(transformer_link)  # type: ignore
            model = model.to(self.device)
            if str(self.device) == "cpu":
                model = model.to(torch.float32)
            model = model.eval()
            tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
            embeddings = self._get_embeddings_core(seqs, model, self.device, tokenizer)
            return embeddings

        def _get_embeddings_core(
            self, seqs: list[str] | numpyArray[np.str_], model, device, tokenizer, chunk_size=None
        ) -> list[list[float]]:
            seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]
            assert tokenizer is not None
            # tokenize sequences and pad up to the longest sequence in the batch
            if chunk_size is None:
                if "cpu" in str(device):
                    chunk_size = 1
                else:
                    chunk_size = 16
            # generate embeddings
            with torch.no_grad():
                res = []
                ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                for i in CustomTqdm(range(0, len(seqs), chunk_size), desc="T5 Embedding Progress", position=1):
                    rng = slice(i, i + chunk_size)
                    input_ids = torch.tensor(ids["input_ids"][rng]).to(device)
                    attention_mask = torch.tensor(ids["attention_mask"][rng]).to(device)
                    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

                    emb_per_proteins = [
                        embedding_repr.last_hidden_state[j][:-1].mean(dim=0) for j in range(len(input_ids))
                    ]

                    embedded = [[float(ee) for ee in e.cpu().numpy().tolist()] for e in emb_per_proteins]
                    res += embedded
            self.outer.new_embs = dict(zip([seq.replace(" ", "") for seq in seqs], res))
            return res


def test():
    sample_seqs = ["ACDEF", "PEPTIDE", "SEQWENCE", "PREWTEIN", "PREWTEEN", "PREWTEEM"]
    t5_embedder = GetT5()
    logger.debug(emb := t5_embedder.get_t5(sample_seqs, device=torch.device("cpu")))
    logger.debug(emb.shape)


if __name__ == "__main__":
    test()
