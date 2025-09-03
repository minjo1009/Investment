
from datetime import datetime, timezone

def blocked(dt_utc, windows_utc):
    for s,e in windows_utc:
        if s<=dt_utc<e: return True
    return False

def test_blocklist_basic():
    now=datetime(2025,1,1,3,0,0,tzinfo=timezone.utc)
    bl=[(datetime(2025,1,1,2,0,0,tzinfo=timezone.utc), datetime(2025,1,1,4,0,0,tzinfo=timezone.utc))]
    assert blocked(now, bl) is True
