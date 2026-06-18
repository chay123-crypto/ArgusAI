import random
import colorsys
from IPython.display import IFrame
import plotly.io as pio
import re
import numpy as np
from cryptography.fernet import Fernet
import pandas as pd
import json
import time
from config import dashboard_palettes,RETRY_MAX_ATTEMPTS,RETRY_BACKOFF,RETRY_TIMEOUT
from functools import wraps

def retry(max_attempts=RETRY_MAX_ATTEMPTS,delay=RETRY_TIMEOUT,backoff=RETRY_BACKOFF,exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            wait=delay
            for attempt in range(max_attempts):
                try:
                    return func(*args,**kwargs)
                except exceptions as e:
                    if attempt==max_attempts-1:
                        raise
                    print(f"[{func.__name__}] attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    wait*=backoff
        return wrapper
    return decorator

def safe_parse(text):
    def unescape(obj):
        if isinstance(obj, str):
            return obj.replace('\\n', '\n').replace('\\t', '\t')
        elif isinstance(obj, dict):
            return {k: unescape(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [unescape(i) for i in obj]
        return obj
    text=re.sub(r'```(?:json)?\s*', '', text).strip()
    try:
        return unescape(json.loads(text))
    except:
        decoder = json.JSONDecoder()
        text = text.strip()
        for start in range(len(text)):
            if text[start] in ('{', '['):
                try:
                    obj, _ = decoder.raw_decode(text, start)
                    return unescape(obj)
                except:
                    continue
        raise ValueError("No valid JSON returned")
    
def sanitize_keys(obj, path=""):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                print(f"🔴 FOUND NON-STRING KEY at {path}: {k!r} (type: {type(k).__name__})")
                new_k = str(k) 
            else:
                new_k = k
            new_dict[new_k] = sanitize_keys(v, f"{path}.{new_k}")
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [sanitize_keys(item, f"{path}[{i}]") for i, item in enumerate(obj)]
    else:
        return obj 

def clean_report(report):
    replacements = {
        '\u202f': ' ',  
        '\u2011': '-',   
        '\u2013': '-',   
        '\u2014': '--', 
        '\u00a0': ' ',  
    }
    for unicode_char, replacement in replacements.items():
        report = report.replace(unicode_char, replacement)
    return report             

def random_template():
    bootstrap_themes=["flatly","darkly","cyborg","cosmo","morph","quartz","vapor","lux","slate","solar"]
    dp=random.choice(list(dashboard_palettes.values()))
    bs=random.choice(bootstrap_themes)
    return dp["plotly"], bs, dp

def derive_accents(base_hex, n=4):
    h=int(base_hex.lstrip('#'), 16)
    r,g,b=(h >> 16) / 255, ((h >> 8) & 0xff) / 255, (h & 0xff) / 255
    hue,sat,val=colorsys.rgb_to_hsv(r, g, b)
    return [
        '#%02x%02x%02x' % tuple(int(c * 255) for c in colorsys.hsv_to_rgb((hue + i / n) % 1, sat, val))
        for i in range(n)
    ]

def clean_results(search_results):
    extracted=[]
    for result in search_results:
        contents=[r['content'] for r in result['response']['results']]
        extracted.append(
            {
                'query':result['query'],
                'content':contents
            }
        )
    return extracted