"""Tiny helper so figure scripts read numbers from analysis output instead of
hard-coding them. Analysis scripts call save(name, obj); figure scripts call load(name).
Files live in ./results/<name>.json relative to the working directory."""
import json, os
import numpy as np

def _clean(o):
    if isinstance(o,dict): return {str(k):_clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_clean(v) for v in o]
    if isinstance(o,(np.floating,float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o,np.integer): return int(o)
    if isinstance(o,np.ndarray): return _clean(o.tolist())
    return o

def save(name,obj):
    os.makedirs("results",exist_ok=True)
    with open(os.path.join("results",name+".json"),"w",encoding="utf-8") as fh:
        json.dump(_clean(obj),fh,indent=1)

def load(name):
    p=os.path.join("results",name+".json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found: run the analysis script that writes it first (see README).")
    with open(p,encoding="utf-8") as fh: return json.load(fh)
