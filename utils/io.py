"""Leaf I/O helpers. No imports toward simulation/fitting/catalog code."""

from pathlib import Path

import pandas as pd


def save_dict_as_parquet(d: dict, path: str | Path, append: bool = True):
    """Guarda un dict (valores tipo lista) como Parquet. Si append=True,
    concatena con el archivo existente (si lo hay) antes de escribir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # crea subdirectorios si no existen

    df_new = pd.DataFrame.from_dict(d)

    if append and path.exists():
        df_old = pd.read_parquet(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_parquet(path, engine="pyarrow", index=False)
