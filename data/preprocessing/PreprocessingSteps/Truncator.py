from typing import Protocol

import pandas as pd
from copy import deepcopy


class Truncator(Protocol):
    max_len: int

    def __init__(self, max_len: int | float):
        assert max_len >= 0
        assert int(max_len) == max_len, '`max_len` cannot have a decimal part.'
        self.max_len = int(max_len)


    def _validate_data(self, seq_list: list[str]):
        for seq in seq_list:
            assert isinstance(seq, str) or pd.isna(seq), f"The sequence {seq} must be a string or {pd.NA}."
            assert len(seq) <= self.max_len, (
                f"The sequence {seq[:10]}{'...' if len(seq) >= 10 else ''} was {len(seq)} characters long but the"
                f" sequence length must be <= {self.max_len}"
            )

    def shrink_data(self, df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
        ...

class DataRemovingTrucator(Truncator):
    def shrink_data(self, df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
        dfdc = deepcopy(df)
        for i, r in dfdc.iterrows():
            if len(r[seq_col]) > self.max_len:
                df.at[i, seq_col] = pd.NA
        self._validate_data(df[seq_col].tolist())
        return df

class DataShorteningTrucator(Truncator):
    def shrink_data(self, df: pd.DataFrame, seq_col: str):
        for i, _ in df.iterrows():
            # print(df.index.tolist()[:10])
            # print(df.columns.tolist()[:10])
            df.at[i, seq_col] = df.at[i, seq_col][:self.max_len]
        self._validate_data(df[seq_col].tolist())
        return df

