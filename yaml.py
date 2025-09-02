import json, re
from typing import Any

def _parse_value(val: str) -> Any:
    val = val.strip()
    if val.startswith('[') and val.endswith(']'):
        inner = val[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(',')]
        return [_parse_value(p) for p in parts]
    if val.startswith('{') and val.endswith('}'):
        inner = val[1:-1].strip()
        if not inner:
            return {}
        out = {}
        for item in inner.split(','):
            k,v = item.split(':',1)
            out[k.strip()] = _parse_value(v.strip())
        return out
    if val.lower() in ('true','false'):
        return val.lower()=='true'
    if val.lower()=='null':
        return None
    try:
        if '.' in val or 'e' in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val.strip('"\'')

def safe_load(stream) -> Any:
    if hasattr(stream, 'read'):
        text = stream.read()
    else:
        text = str(stream)
    lines = []
    for line in text.splitlines():
        line = re.sub(r'#.*', '', line).rstrip()
        if line:
            lines.append(line)
    tokens = []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        tokens.append((indent, line.lstrip()))
    idx = 0
    def parse_block(indent):
        nonlocal idx
        if idx >= len(tokens):
            return None
        if tokens[idx][0] != indent:
            return None
        # list or dict
        if tokens[idx][1].startswith('- '):
            arr = []
            while idx < len(tokens) and tokens[idx][0] == indent and tokens[idx][1].startswith('- '):
                line = tokens[idx][1][2:].strip()
                idx += 1
                if idx < len(tokens) and tokens[idx][0] > indent:
                    idx -= 1
                    val = parse_block(indent+2)
                else:
                    val = _parse_value(line)
                arr.append(val)
            return arr
        else:
            d = {}
            while idx < len(tokens) and tokens[idx][0] == indent:
                line = tokens[idx][1]
                if not line or ':' not in line:
                    idx += 1
                    continue
                key, val = line.split(':',1)
                key = key.strip()
                val = val.strip()
                idx += 1
                if idx < len(tokens) and tokens[idx][0] > indent and val=="":
                    val = parse_block(indent+2)
                elif val=="":
                    val = None
                else:
                    val = _parse_value(val)
                d[key] = val
            return d
    result = parse_block(0)
    return result

def safe_dump(data: Any, stream) -> None:
    json.dump(data, stream, indent=2)
