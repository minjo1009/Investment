import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from backtest.utils.dedupe import dedupe_columns, safe_load_no_dupe


def test_dedupe_columns():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    out = dedupe_columns(df)
    assert list(out.columns) == ["a"]
    assert out.attrs["_dedupe_info"]["dropped_columns"] == ["a"]


def test_safe_load_no_dupe_error():
    yaml_str = "a: 1\na: 2\n"
    with pytest.raises(ValueError):
        safe_load_no_dupe(io.StringIO(yaml_str))
