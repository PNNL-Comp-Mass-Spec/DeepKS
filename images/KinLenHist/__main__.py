"""Main entry point for running KinLenHist."""

from ...config.join_first import join_first
import pandas as pd
from .KinLenHist import n_tile_hist


df = pd.read_csv(join_first("data/raw_data/kinase_seq_918.csv", 2, __file__))
seqs = df['kinase_seq'].values.tolist()
data = [len(seq) for seq in seqs]

n_tile_hist(data, n_tile=4, total_bins=600)