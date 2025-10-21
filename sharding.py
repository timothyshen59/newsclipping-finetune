#!/usr/bin/env python3
"""
Sharding pipeline for VisualNews / NewsCLIPpings dataset.
"""

import os
import io
import json
import logging
import base64

from pathlib import Path
from typing import Dict, Any
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import webdataset as wds
import argparse
import pandas as pd 

def setup_logger(log_file: Path) -> None : 
    handlers = [logging.StreamHandler()]
    if log_file: 
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers = handlers,
    )

def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 90) -> bytes: 
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality = quality)
    
    return buf.getvalue()


def shard_dataset(
    data_path: Path,
    root_dir: Path, 
    out_dir: Path,
    preprocessed: bool = False,
    samples_per_shard: int = 5000,
    quality: int = 90,    
    parquet_flush_amount: int = 50000,
) -> None : 
    
    if preprocessed:
        df = pd.read_parquet(data_path)
        data = df.to_dict(orient="records")
    else: 
        with open(data_path, "r") as f: 
            data = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_records = []

    shard_idx, in_shard= 0,0 
    sink = wds.TarWriter(str(out_dir/ f"shard-{shard_idx:06d}.tar"))
    
    
    for i,ex in enumerate(tqdm(data, desc="Sharding")): 
        if preprocessed:
            img_bytes = base64.b64decode(ex["image_b64"])
            article_txt = ex["text"]
        else: 
            
            img_path = root_dir / ex["image_path"]
            txt_path = root_dir / ex["article_path"]
            
            if not img_path.exists() or not txt_path.exists(): 
                logging.warning(f"Missing file for ID {ex['id']}. Skipping.")
                continue 
        
            try: 
                img = Image.open(img_path)
                img_bytes = pil_to_jpeg_bytes(img, quality=quality)
            except (UnidentifiedImageError, OSError) as e: 
                logging.error(f"Image load failed for {img_path}: {e}")
                continue 
        
            try: 
                with open(txt_path, "r", encoding="utf-8",errors="ignore") as ftxt: 
                    article_txt = ftxt.read() 
            except Exception as e: 
                logging.error(f"Text load failed for {txt_path}: {e}")   
                continue
            
        key = f"{ex['source']}_{ex['id']}"
        shard_name = f"shard-{shard_idx:06d}.tar"
        
        
        sample: Dict[str, Any] = { 
            "__key__": key, 
            "jpg": img_bytes, 
            "txt": article_txt, 
            "json": { 
                "id": ex['id'],
                "source": ex['source'],
                "topic": ex['topic'],
                "caption": ex['caption'],    
            },                                
                                  
        }
        
        idx_record = {
            "id": ex["id"],
            "source": ex["source"],
            "topic": ex["topic"],
            "caption": ex["caption"],
            "key": key,
            "shard": shard_name,
        }
        
        index_records.append(idx_record)
        sink.write(sample)
    
        in_shard += 1 
        
        if in_shard >= samples_per_shard: 
            sink.close() 
            shard_idx += 1 
            sink = wds.TarWriter(str(out_dir / f"shard-{shard_idx:06d}.tar"))
            in_shard = 0

        if (i + 1) % parquet_flush_amount == 0: 
            df = pd.DataFrame(index_records)
            parquet_path = out_dir / f"index-{i+1:09d}.parquet"
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            logging.info(f"Flushed Parquet Index {i+1} to {parquet_path}")
            index_records = []
            
    if index_records:
        df = pd.DataFrame(index_records)
        parquet_path = out_dir / f"index-final.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        logging.info(f"Flushed final Parquet index to {parquet_path}")

    sink.close() 
    logging.info(f"Finished sharding into {shard_idx + 1} shards at {out_dir}")
    

def parse_args(): 
    parser = argparse.ArgumentParser(description="Shard VisualNews into WebDataset Archieves")
    parser.add_argument("--data-json", type=Path, required=True, help="Path to data.json")
    parser.add_argument("--root-dir", type=Path, required=True, help="Path to expanded VisualNews tar file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for shards")
    
    parser.add_argument("--samples-per-shard", type=int, default=5000, help="Samples per shard")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    parser.add_argument("--log-file", type=Path, help="Optional log file path")
    
    return parser.parse_args()



def main(): 
    args = parse_args() 
    setup_logger(args.log_file)
    shard_dataset(
        data_json=args.data_json,
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        samples_per_shard=args.samples_per_shard,
        quality=args.quality,
    )
    

if __name__ == "__main__":
    main()






