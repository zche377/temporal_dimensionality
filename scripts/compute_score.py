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
from tqdm.auto import tqdm

from lib.analyses.main_analyses import *
from lib.analyses.weights_sanity_checks import *
from lib.analyses.model_analyses import *
from lib.datasets import load_n_subjects, load_dataset

import xarray as xr
import numpy as np
from bonner.caching import (
    BONNER_CACHING_HOME,
    cache,
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # analysis related
    parser.add_argument("--script", type=str, default=None)
    parser.add_argument("--analysis", type=str, default="behavior")
    # subject related
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--nyield", type=int, default=9999)
    # data related
    parser.add_argument("--dataset", type=str, default="things_eeg_2_test")
    parser.add_argument("--freq", type=int, default=250)
    # scorer related
    parser.add_argument("--model", type=str, default="linear")
    # weights_analysis related
    parser.add_argument("--space", type=str, default=None)
    parser.add_argument("--dim1", type=str, default=None)
    parser.add_argument("--dim2", type=str, default=None)
    parser.add_argument("--dim3", type=str, default=None)
    parser.add_argument("--dim4", type=str, default=None)
    parser.add_argument("--subset", dest="subset", action="store_true")
    parser.set_defaults(subset=False)
    parser.add_argument("--pca", dest="pca", action="store_true")
    parser.set_defaults(pca=False)
    parser.add_argument("--repc", dest="repc", action="store_true")
    parser.set_defaults(repc=False)
    parser.add_argument("--ssm", type=str, default="cumulative")
    parser.add_argument("--pctime", type=float, default=.09)
    parser.add_argument("--time", type=str, default="significant")
      # model_analysis related
    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # split_dim_weights related
    parser.add_argument("--splitdim", type=str, default=None)
    parser.add_argument("--nsplits", type=int, default=2)
    parser.add_argument("--rois", type=str, default=None)
    
    args = parser.parse_args()
    
    meg_ldk = {
        ### fixed for now for things meg ###
        "from_raw": 1,
        "downsample_freq": 100,
        "h_freq": 100,
        "baseline": (None, 0),
        "rois": 'op',
        # "rois": 'o',
        # "rois": 'f',
        ### options ###
    }
    eeg_ldk = {
        ### fixed for now for things meg ###
        "from_raw": 1,
        "rois": 'all',
        ### options ###
    }
    ldk_dict = {
        # TEMP
        "things_eeg_2": {},
        "things_eeg_2_train": {},
        "things_eeg_2_test": {},
        "things_meg": meg_ldk,
        "things_meg_train": meg_ldk,
        "things_meg_test": meg_ldk,
    }
    load_dataset_kwargs = ldk_dict[args.dataset]
    if args.rois is not None:
        load_dataset_kwargs["rois"] = args.rois
    
    scorer_kwargs={
        ### fixed for now ###
        "model_name": args.model,
        #####################
    }
    
    match args.model:
        case "lda":
            scorer_kwargs["shrinkage"] = 1e-2
        case "linear":
            scorer_kwargs["l2_penalty"] = 1e-2
        case "ridge":
            pass
        case _:
            pass
    
    assert args.script is not None
    match args.script:
        # validated
        case "decoding":
            fn = decoding
            loop_subject = True
            kwargs = {}
        case "generalization":
            fn = decoding_generalization
            loop_subject = True
            kwargs = {}
        case "pc_generalization":
            fn = pc_generalization
            loop_subject = False
            kwargs = {
                "space": args.space,
                "pc_dims": (args.dim1, args.dim2),
                "generalization_dim": args.dim3,
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "split_dim_weights":
            fn = split_dim_weights
            loop_subject = False
            kwargs = {
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "kernel_comparison":
            fn = kernel_comparison
            loop_subject = False
            kwargs = {
                "space": args.space,
                "kernel_dims": (args.dim1, args.dim2),
                "comparison_dim": args.dim3,
                "split_dim": args.splitdim,
            }
        case "kernel_comparison_cs":
            fn = kernel_comparison_cross_subject
            loop_subject = False
            kwargs = {
                "space": args.space,
                "kernel_dims": (args.dim1, args.dim2),
                "comparison_dim": args.dim3,
                "split_dim": args.splitdim,
            }
        case "geometry":
            fn = geometries
            loop_subject = False
            kwargs = {
                "space": args.space,
                "geo_dims": (args.dim1, args.dim2),
                "list_dim": args.dim3,
            }
        case "dr_vis":
            fn = dim_reduction_visualization
            loop_subject = False
            kwargs = {
                "space": args.space,
                "reduce_dims": (args.dim1, args.dim2),
                "list_dim": args.dim3,
            }
        case "kernels":
            fn = kernels
            loop_subject = False
            kwargs = {
                "space": args.space,
                "kernel_dims": (args.dim1, args.dim2),
                "list_dim": args.dim3,
                "split_dim": args.splitdim,
            }
        case "ttpca":
            fn = tt_pca
            loop_subject = False
            kwargs = {
                "space": args.space,
                "pc_dims": (args.dim1, args.dim2),
                "generalization_dim": args.dim3,
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "ttpca_nonclustered":
            fn = tt_pca_nonclustered
            loop_subject = False
            kwargs = {
                "space": args.space,
                "pc_dims": (args.dim1, args.dim2),
                "generalization_dim": args.dim3,
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "ttpcacs":
            fn = tt_pca_cross_subject_splits
            loop_subject = False
            kwargs = {
                "space": args.space,
                "pc_dims": (args.dim1, args.dim2),
                "generalization_dim": args.dim3,
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "ttspca":
            fn = tt_stack_pca
            loop_subject = False
            kwargs = {
                "space": args.space,
                "pc_dims": (args.dim1, args.dim2),
                "stack_dim": args.dim3,
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
                "time": args.time,
            }
            
        # validating
        case "rmv":
            fn = remaining_var
            loop_subject = False
            kwargs = {
                "pca": args.pca,
                "target_timepoint": args.pctime,
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "rmvc":
            fn = remaining_var_concat
            loop_subject = False
            kwargs = {
                "pca": args.pca,
                "target_timepoint": args.pctime,
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "pcr":
            fn = umap_pc_reconstruction
            loop_subject = False
            kwargs = {
                "target_timepoint": args.pctime,
            }
        case "plssvdcs":
            fn = plssvd_cross_subject
            loop_subject = False
            kwargs = {}
        case "ttd":
            fn = tt_decoding
            loop_subject = False
            kwargs = {
                "subset": args.subset,
                "pca": args.pca,
                "reconstruct_with_pc": args.repc,
                "subset_method": args.ssm,
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "ttdcs":
            fn = tt_decoding_cross_subject
            loop_subject = False
            kwargs = {
                "subset": args.subset,
                "pca": args.pca,
                "reconstruct_with_pc": args.repc,
                "subset_method": args.ssm,
                "pc_timepoint": args.pctime,
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "ttsd":
            fn = tt_stack_decoding
            loop_subject = True
            kwargs = {
                "subset": args.subset,
                "pca": args.pca,
                "stack_dims": (args.dim1, args.dim2),
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "tte":
            fn = tt_encoding
            loop_subject = True
            kwargs = {
                "subset": args.subset,
                "pca": args.pca,
                "reconstruct_with_pc": args.repc,
                "subset_method": args.ssm,
                "pc_timepoint": args.pctime,
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "ttdgen":
            fn = tt_decoding_generalization
            loop_subject = False
            kwargs = {
                "subset": args.subset,
                "pca": args.pca,
                "reconstruct_with_pc": args.repc,
                "subset_method": args.ssm,
                "pc_timepoint": args.pctime,
                "model_kwargs": {
                    "model_uid": args.uid
                }
            }
        case "sdwr":
            fn = split_dim_weights_rdm
            loop_subject = False
            kwargs = {
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "sdwrcs":
            fn = split_dim_weights_rdm_cross_subjects
            loop_subject = False
            kwargs = {
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "sdwcs":
            fn = split_dim_weights_cross_subjects
            loop_subject = False
            kwargs = {
                "split_dim": args.splitdim,
                "n_splits": args.nsplits,
            }
        case "mftma":
            fn = mftma_geometries
            loop_subject = False
            kwargs = {
                "space": args.space,
                "geo_dims": (args.dim1, args.dim2),
                "list_dim": args.dim3,
                "loop_dim": args.dim4,
            }
        case "rot_inv":
            fn = rot_inv_generalization
            loop_subject = True
            kwargs = {}
        case "preproc":
            fn = None
            for subject in range(args.sidx+1, min(args.sidx+args.nyield+1, load_n_subjects(args.dataset)+1)):
                data = load_dataset(
                    args.dataset,
                    subjects=subject,
                    **load_dataset_kwargs,
                )
        case "weights_eeg_vp":
            fn = weights_eeg_variance_partitioning
            loop_subject = False
            kwargs = {}
        case "model_decoding":
            fn = model_decoding
            loop_subject = False
            kwargs = {
                "model_uid": args.uid,
                "seed": args.seed,
            }
        case _:
            raise ValueError(f"Unknown script: {args.script}")
        
    
    if fn is not None:
        if not loop_subject:
            fn(
                analysis=args.analysis,
                dataset=args.dataset,
                load_dataset_kwargs=load_dataset_kwargs,
                scorer_kwargs=scorer_kwargs,
                **kwargs,
            )
        else:
            for subject in tqdm(range(args.sidx+1, min(args.sidx+args.nyield+1, load_n_subjects(args.dataset)+1)), desc="subject"):
                logging.info(f"subject={subject}")
                fn(
                    analysis=args.analysis,
                    dataset=args.dataset,
                    subject=subject,
                    load_dataset_kwargs=load_dataset_kwargs,
                    scorer_kwargs=scorer_kwargs,
                    **kwargs,
                )
    



