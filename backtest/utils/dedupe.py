import pandas as pd
import importlib.machinery, importlib.util, sys, os, re

_cwd = os.path.abspath(os.getcwd())
_search_paths = [p for p in sys.path if p and os.path.abspath(p) != _cwd]
_spec = importlib.machinery.PathFinder.find_spec('yaml', _search_paths)
if _spec is None:
    import yaml as _yaml  # fallback, may be local
else:
    _yaml = importlib.util.module_from_spec(_spec)
    sys.modules['yaml'] = _yaml
    _spec.loader.exec_module(_yaml)
yaml = _yaml


def dedupe_columns(df: pd.DataFrame, keep: str="first") -> pd.DataFrame:
    dup = df.columns.duplicated(keep=keep)
    if dup.any():
        dropped = list(df.columns[dup])
        df = df.loc[:, ~dup].copy()
        meta = df.attrs.get("_dedupe_info", {})
        meta["dropped_columns"] = meta.get("dropped_columns", []) + dropped
        df.attrs["_dedupe_info"] = meta
    return df


if all(hasattr(yaml, attr) for attr in ("SafeLoader", "load", "resolver")):
    class DuplicateKeyLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader, node, deep=False):
        mapping = {}
        for k_node, v_node in node.value:
            k = loader.construct_object(k_node, deep=deep)
            if k in mapping:
                raise ValueError(f"YAML duplicate key detected: {k!r}")
            mapping[k] = loader.construct_object(v_node, deep=deep)
        return mapping

    DuplicateKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
    )

    def safe_load_no_dupe(stream):
        return yaml.load(stream, Loader=DuplicateKeyLoader)
else:
    def safe_load_no_dupe(stream):
        if hasattr(stream, "read"):
            text = stream.read()
        else:
            text = str(stream)

        lines = []
        for raw_line in text.splitlines():
            line = re.sub(r"#.*", "", raw_line).rstrip()
            if line:
                indent = len(line) - len(line.lstrip())
                lines.append((indent, line.lstrip()))

        stack = [( -1, set())]
        for indent, line in lines:
            while stack and indent <= stack[-1][0]:
                stack.pop()
            if ":" in line:
                key, rest = line.split(":", 1)
                key = key.strip()
                if key in stack[-1][1]:
                    raise ValueError(f"YAML duplicate key detected: {key!r}")
                stack[-1][1].add(key)
                if not rest.strip():
                    stack.append((indent, set()))

        return yaml.safe_load(text)
