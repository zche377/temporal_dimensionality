from dotenv import load_dotenv
load_dotenv()

import sys
import os
from pathlib import Path
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

import logging
logging.basicConfig(level=logging.INFO)
import argparse

from lib.datasets import load_dataset, load_n_subjects
from lib.models import DJModel

# from bonner.datasets.gifford2022_things_eeg_2 import download_dataset
from bonner.datasets.zhang2025_nod_eeg import download_dataset
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--nyield", type=int, default=9999)
    parser.add_argument("--dataset", type=str, default="things_eeg_2_train")
    
    parser.add_argument("--mid", type=str, default="openclip_rn50_yfcc15m")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    # model = DJModel(args.mid, seed=args.seed)
    
    
    # _ = model(args.dataset, dataloader_kwargs={"batch_size": 32})
    
    download_dataset()
    
    # for subject in range(args.sidx+1, min(args.sidx+args.nyield+1, load_n_subjects(args.dataset)+1)):
    #     logging.info(subject)
    #     load_dataset(args.dataset, subjects=subject)
    
    
    
