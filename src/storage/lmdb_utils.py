from pathlib import Path

import lmdb
import streamlit as st


@st.cache_resource()
def get_lmdb_env(path: str | Path, max_dbs: int, map_size: int) -> lmdb.Environment:
    """Create cached LMDB environment."""
    return lmdb.Environment(
        path=str(path),
        max_dbs=max_dbs,
        map_size=map_size,
    )
