from pathlib import Path

import streamlit as st
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template


@st.cache_resource()
def get_j2_template(searchpath: str | Path, name: str) -> Template:
    """Load jinja2 template with strict undefined behavior."""
    searchpath = Path(searchpath)
    env = Environment(
        loader=FileSystemLoader(searchpath=searchpath),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(name=name)
