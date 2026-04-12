#!/usr/bin/env python3
"""
Step 1: Generate the full 75-pair mapping from SONICS metadata.
Downloads real songs from YouTube and AI songs from HuggingFace zips.
"""

import os
import json
import subprocess
import time
import re
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from remotezip import RemoteZip

# ── Paths ─────────────────────────────────────────────────────────────────────
PAIRS_DIR = Path("/home/nikos_pc/Desktop/Orfium/orfium_audio_attribution_task/qwen/results/real_ai_pairs")
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Build the full mapping ────────────────────────────────────────────

print("=== Step 1: Building full 75-pair mapping ===")

path_fake = hf_hub_download('awsaf49/sonics', 'fake_songs.csv', repo_type='dataset')
df_fake = pd.read_csv(path_fake, dtype={'source': str, 'target': str}, low_memory=False)
path_real = hf_hub_download('awsaf49/sonics', 'real_songs.csv', repo_type='dataset')
df_real = pd.read_csv(path_real)

def lyrics_key(text):
    if pd.isna(text):
        return ''
    return hash(str(text)[:500])

df_fake['lyrics_hash'] = df_fake['lyrics'].apply(lyrics_key)
df_real['lyrics_hash'] = df_real['lyrics'].apply(lyrics_key)

real_by_hash = {}
for _, row in df_real.iterrows():
    h = row['lyrics_hash']
    if h and h not in real_by_hash:
        real_by_hash[h] = {
            'title': row['title'],
            'artist': row['artist'],
            'youtube_id': row['youtube_id'],
            'real_filename': row['filename'],
        }

# Build mapping: fake_id → {real_info, fake_files}
pairs = {}
for _, row in df_fake.iterrows():
    h = row['lyrics_hash']
    if h in real_by_hash:
        fid = int(row['id'])
        if fid not in pairs:
            ri = real_by_hash[h]
            pairs[fid] = {
                'real_title': ri['title'],
                'real_artist': ri['artist'],
                'youtube_id': ri['youtube_id'],
                'algorithm': row['algorithm'],
                'fake_files': [],
            }
        pairs[fid]['fake_files'].append(row['filename'])

print(f"Found {len(pairs)} ground truth real→AI pairs")

# Save mapping
mapping_path = PAIRS_DIR / "real_ai_pairs_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(pairs, f, indent=2)
print(f"Mapping saved to {mapping_path}")

# ── Step 2: Download AI songs from HuggingFace ────────────────────────────────

print("\n=== Step 2: Downloading AI songs from HuggingFace ===")

path_meta = hf_hub_download('awsaf49/sonics', 'metadata.json', repo_type='dataset')
with open(path_meta) as f:
    meta = json.load(f)

# Build reverse mapping: fake_filename → zip_part
file_to_zip = {fn.split('/')[-1]: zp for fn, zp in meta['file_mapping'].items()}

downloaded_ai = 0
for fid, info in pairs.items():
    for fake_fn in info['fake_files']:
        basename = fake_fn.split('/')[-1]
        local = PAIRS_DIR / f"fake_{fid}_{basename.split('_')[-2]}_{basename.split('_')[-1]}"
        if local.exists():
            continue
        
        zip_part = file_to_zip.get(basename)
        if not zip_part:
            print(f"  WARNING: {basename} not found in file_mapping")
            continue
        
        url = f"https://huggingface.co/datasets/awsaf49/sonics/resolve/main/{zip_part}"
        try:
            rz = RemoteZip(url)
            data = rz.read(f"fake_songs/{basename}")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            downloaded_ai += 1
            rz.close()
        except Exception as e:
            print(f"  ERROR downloading {basename}: {e}")

print(f"Downloaded {downloaded_ai} AI songs")

# ── Step 3: Download real songs from YouTube ──────────────────────────────────

print("\n=== Step 3: Downloading real songs from YouTube ===")

downloaded_real = 0
skipped_real = 0
failed_real = []

for fid, info in pairs.items():
    yt_id = info['youtube_id']
    local = PAIRS_DIR / f"real_{fid}.mp3"
    
    if local.exists():
        skipped_real += 1
        continue
    
    url = f"https://www.youtube.com/watch?v={yt_id}"
    cmd = [
        'yt-dlp', '--js-runtimes', 'node',
        '-x', '--audio-format', 'mp3', '--audio-quality', '192K',
        '-o', str(local),
        '--socket-timeout', '30',
        '--retries', '3',
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and local.exists():
            downloaded_real += 1
            print(f"  ✓ [{downloaded_real+skipped_real}/{len(pairs)}] {info['real_title']} by {info['real_artist']}")
        else:
            print(f"  ✗ [{downloaded_real+skipped_real}/{len(pairs)}] {info['real_title']} - exit code {result.returncode}")
            failed_real.append((fid, info['real_title'], info['real_artist'], yt_id))
            if local.exists():
                local.unlink()
    except subprocess.TimeoutExpired:
        print(f"  ⏱ [{downloaded_real+skipped_real}/{len(pairs)}] {info['real_title']} - timeout")
        failed_real.append((fid, info['real_title'], info['real_artist'], yt_id))
    except Exception as e:
        print(f"  ✗ [{downloaded_real+skipped_real}/{len(pairs)}] {info['real_title']} - {e}")
        failed_real.append((fid, info['real_title'], info['real_artist'], yt_id))
    
    # Rate limit
    time.sleep(1)

print(f"\nDownloaded: {downloaded_real}, Skipped (already exist): {skipped_real}, Failed: {len(failed_real)}")

if failed_real:
    failed_path = PAIRS_DIR / "failed_downloads.json"
    with open(failed_path, 'w') as f:
        json.dump([{
            'fid': fid, 'title': t, 'artist': a, 'youtube_id': y
        } for fid, t, a, y in failed_real], f, indent=2)
    print(f"Failed downloads saved to {failed_path}")

# Summary
real_files = list(PAIRS_DIR.glob("real_*.mp3"))
ai_files = list(PAIRS_DIR.glob("fake_*.mp3"))
print(f"\n=== SUMMARY ===")
print(f"Real songs downloaded: {len(real_files)}")
print(f"AI songs downloaded: {len(ai_files)}")
print(f"Complete pairs (both real + at least one AI): ", end="")
complete = 0
for fid in pairs:
    real_exists = (PAIRS_DIR / f"real_{fid}.mp3").exists()
    ai_exists = any(f.name.startswith(f"fake_{fid}_") for f in ai_files)
    if real_exists and ai_exists:
        complete += 1
print(f"{complete}")
