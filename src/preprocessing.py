#!/usr/bin/env python3

"""
NewsCLIPpings Preprocessing Script

Usage: 
python preprocessing.py \
        --data-json path/to/data.json \
        --root-dir path/to/images/ \
        --out-dir path/to/output_dir \
        --split-json path/to/split.json \
        --split-str "train"/"test"/"validate" 
"""


import re 
import logging 
import argparse 
import ray 
import ray.data as rd
import io 
import base64

from pathlib import Path 
from typing import Dict, Any, Optional, Iterable, Iterator, List, TypeVar

from itertools import islice

from PIL import Image, UnidentifiedImageError
from src.ingestion import iter_entries 


logger = logging.getLogger(__name__)

def setup_logger():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def chunked_iter(it: Iterator , size: int = 2000) -> Iterator[List]: 
    iterator = iter(it)
    
    while True: 
        batch = list(islice(iterator,size))

        if not batch: 
            break
        
        yield batch 
    
    
def clean_text(text: str) -> str: 
    text = text.lower().strip() 
    text = re.sub(r"\s+", " ", text)
    return text 

def preprocess_image(img_path: Path, size: int = 224) -> Optional[Image.Image]: 
    try: 
        img = Image.open(img_path)

        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA").convert("RGB")
            
        elif img.mode != "RGB":
            img = img.convert("RGB")
            
        img = img.resize((size,size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality = 90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    

    except (UnidentifiedImageError, OSError) as e: 
        logger.error(f"Image load failed for {img_path}: {e}")
        return None 

def preprocess_entry(entry: Dict[str, Any], img_size: int = 224) -> Optional[Dict[str,Any]]: 

    output = {
        "caption": clean_text(entry["caption"]),
        "label": entry["label"],
    }

    img = preprocess_image(entry["img_path"], size=img_size)
    
    if img is None: 
        return output

    output["image"] = img
    output["valid"] = True
    
    return output
    
def parse_args(): 
    parser = argparse.ArgumentParser(description="Preprocess VisualNews dataset")
    parser.add_argument("--data-json", type=Path, required=True, help="data.json path")
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory path")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for preprocessing")
    parser.add_argument("--split-json", type=Path, required=True,help="Path to target split json (train, test, validate)")
    parser.add_argument("--split-str", type=str, required=True,help="String you want to label split (train, test, validate)")

    parser.add_argument("--img-size", type=int, default=224, help="Image resize target")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (None = local)")
    
    return parser.parse_args()


def main(): 
    
    args = parse_args() 
    setup_logger() 
    
    ray.init(address=args.ray_address) 
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    entries = iter_entries(args.data_json, args.root_dir, args.split_json, args.split_str)
    shard_idx = 0 
    
    for batch in chunked_iter(entries): 
        
        ds = rd.from_items(batch)
        processed_ds = ds.map(lambda ex: preprocess_entry(ex, img_size = args.img_size)).filter(expr="valid == True")
        
        parquet_path = args.out_dir / f"preprocessed-shard-{shard_idx:05d}.parquet"
        processed_ds.repartition(1).write_parquet(str(parquet_path))

        logger.info(f"Wrote shard {shard_idx} => {parquet_path}")
        shard_idx += 1 

    
    logger.info(f"Preprocessed {shard_idx} shards to {args.out_dir}")


if __name__ == "__main__":
    main()