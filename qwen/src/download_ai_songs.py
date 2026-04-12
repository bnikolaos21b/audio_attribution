#!/usr/bin/env python3
"""
Download AI songs from HuggingFace zips.
Constructs filenames from IDs directly since fake_files column lacks .mp3 extension.
"""

import json
import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
from remotezip import RemoteZip
from collections import defaultdict

PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")

# Load mapping
mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path) as f:
    pairs = json.load(f)

# Load metadata
path_meta = hf_hub_download('awsaf49/sonics', 'metadata.json', repo_type='dataset')
with open(path_meta) as f:
    meta = json.load(f)

# Build reverse mapping: full_path → zip_part
file_to_zip = {fn: zp for fn, zp in meta['file_mapping'].items()}
url_base = 'https://huggingface.co/datasets/awsaf49/sonics/resolve/main/'

# For each pair, construct the correct filename
# The fake_files column has entries like "fake_20679_suno_0"
# The actual filename in the zip is "fake_songs/fake_20679_suno_0.mp3"
print(f"Total pairs: {len(pairs)}")

downloaded = 0
skipped = 0
errors = 0
not_found = 0

# Group by zip part
zip_groups = defaultdict(list)

for fid, info in pairs.items():
    for fake_base in info['fake_files']:
        # Construct full path
        full_path = f"fake_songs/{fake_base}.mp3"
        
        if full_path in file_to_zip:
            zip_part = file_to_zip[full_path]
            basename = f"{fake_base}.mp3"
            local = PAIRS_DIR / f"ai_{fid}_{basename}"
            zip_groups[zip_part].append((fid, full_path, basename, local))
        else:
            print(f"  NOT FOUND: {full_path}")
            not_found += 1

print(f"Files found in mapping: {sum(len(v) for v in zip_groups.values())}")
print(f"Files not found: {not_found}")
print(f"Zip parts needed: {len(zip_groups)}")

# Download
for zip_part, files in sorted(zip_groups.items()):
    url = url_base + zip_part
    print(f"\n── {zip_part} ({len(files)} files) ──")
    
    try:
        rz = RemoteZip(url)
        
        for fid, full_path, basename, local in files:
            if local.exists():
                skipped += 1
                continue
            
            if full_path in rz.namelist():
                try:
                    data = rz.read(full_path)
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.write_bytes(data)
                    downloaded += 1
                    print(f"  ✓ {basename} ({len(data)/1e6:.1f}MB)")
                except Exception as e:
                    print(f"  ✗ {basename}: {e}")
                    errors += 1
            else:
                print(f"  ✗ {basename}: not in zip")
                errors += 1
        
        rz.close()
    except Exception as e:
        print(f"  ERROR: {e}")
        errors += len(files)
    
    time.sleep(0.5)

print(f"\n=== SUMMARY ===")
print(f"Downloaded: {downloaded}")
print(f"Skipped: {skipped}")
print(f"Errors: {errors}")
print(f"Not found in mapping: {not_found}")

ai_files = list(PAIRS_DIR.glob("ai_*.mp3"))
real_files = list(PAIRS_DIR.glob("real_*.mp3"))
print(f"Real MP3s: {len(real_files)}")
print(f"AI MP3s: {len(ai_files)}")

complete = 0
for fid in pairs:
    real_exists = (PAIRS_DIR / f"real_{fid}.mp3").exists()
    ai_exists = any(f.name.startswith(f"ai_{fid}_") for f in ai_files)
    if real_exists and ai_exists:
        complete += 1
print(f"Complete pairs: {complete}/{len(pairs)}")
