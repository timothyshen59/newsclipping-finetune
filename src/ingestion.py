#!/usr/bin/env python3

"""
NewsClippings Data Ingestion Script

Usage: 
python ingestion.py \
        --data-json path/to/data.json \
        --root-dir path/to/images/ \
        --split-json path/to/split.json \
        --split-str "train"/"test"/"validate" 
"""

import json 
import logging 
import argparse 
import fsspec
import math 

from pathlib import Path 
from typing import Dict, Any, Generator, Optional 

logger = logging.getLogger(__name__)

def load_json(path: Path) -> Any: 
    try: 
        with fsspec.open(path, "r", encoding="utf-8") as f: 
            return json.load(f) 
        
    except Exception as e: 
        logger.error(f"Failed to load JSON {path}: {e}")
        raise 
    
    
def load_json_as_dict(path: Path) -> Dict[str, Dict[str,str]]: 
    try: 
        with fsspec.open(path, "r", encoding="utf-8") as f: 
            entries = json.load(f) 
            
            if isinstance(entries, dict): 
                entries = [entries]
                
            return { 
                    entry["id"]: {
                        "caption": entry.get("caption"), 
                        "image_path": entry.get("image_path")
                    } 
                    for entry in entries
                }
        
    except Exception as e: 
        logger.error(f"Failed to load JSON {path}: {e}")
        raise 
    
    
def normalize_entry(caption_id, image_id, root_dir: Path,  data_dict: Dict[str,Any], label: bool, split_str: str) -> Optional[Dict[str,Any]]: 
    """
    """    
    caption_entry = data_dict.get(caption_id)
    image_entry = data_dict.get(image_id) 
    
    if not caption_entry or not image_entry: 
        logger.warning(f"Missing entries for caption ID {caption_id}, image ID {image_id}")
    
    img_path = root_dir / image_entry["image_path"]
    
    if not img_path.exists(): 
        logger.warning(f"Missing image files for ID {image_id}, skipping...")
        return None 
    
    return { 
        "caption": caption_entry["caption"],
        "img_path": str(img_path), 
        "label": label, 
        "split": split_str,
    }


def iter_entries(data_json: Path, root_dir: Path, split_json: Path, split_str: str) -> Generator[Dict[str,Any], None, None]: 
    data_dict = load_json_as_dict(data_json)
    
    split_labels = load_json(split_json) 
    annotations = split_labels.get("annotations", [])
    score_dist = {} #### 
    
    for entry in annotations:
        caption_id = entry.get("id")
        image_id = entry.get("image_id")
        label = entry.get("falsified")
        score = entry.get("similarity_score") #### 
        
   
        score_dist[math.floor(score / .10)] = score_dist.get(math.floor(score / .10), 0) + 1 ###
        

        norm = normalize_entry(caption_id, image_id, root_dir, data_dict, label, split_str)

        if norm: 
         
            yield norm
            
    print(score_dist)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="VisualNews Data Ingestion")
    parser.add_argument("--data-json", type=Path, required=True,help="Path to data.json")
    parser.add_argument("--root-dir", type=Path, required=True,help="Path to root directory")
    parser.add_argument("--split-json", type=Path, required=True,help="Path to target split json (train, test, validate)")
    parser.add_argument("--split-str", type=str, required=True,help="String you want to label split")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    count = sum(1 for _ in iter_entries(args.data_json, args.root_dir, args.split_json, args.split_str))
    logger.info(f"Ingested {count} entries.")
        

    

