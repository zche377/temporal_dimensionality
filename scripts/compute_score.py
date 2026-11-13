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
    parser.add_argument("--l2", type=float, default=1e-2)
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
    parser.add_argument("--regress_out", type=str, default=None)
    parser.add_argument("--sessions", type=str, default=None)  # ttpca: test presentation(≈session) subset, e.g. "0-5"
    parser.add_argument("--pca_source", type=str, default="train")  # ttpca: "train" basis (default) or "test" (within-test cvPCA)
    parser.add_argument("--fit_source", type=str, default="train")  # ttd: "train" (default) or "test_split" (fit on one test split, eval the other)
    parser.add_argument("--test_split_half", type=int, default=None)  # ttd: restrict TEST to one rep-half (0/1) to SNR-match ttpca
    parser.add_argument("--no_null", dest="no_null", action="store_true")
    parser.set_defaults(no_null=False)
    # Stringer per-image half-rep mode (ttpca + ttd): seed-averaged, in-lib significance, drop the raw
    # null. Default off -> published caches byte-identical (adds .split_mode=stringer.n_seeds= tag).
    parser.add_argument("--per_image", dest="per_image", action="store_true")
    parser.set_defaults(per_image=False)
    parser.add_argument("--n_seeds", type=int, default=1)
    # drop Spearman-Brown (ttpca only): adds .sb=0 tag. Default keeps SB (published behaviour).
    parser.add_argument("--no_sb", dest="spearman_brown", action="store_false")
    parser.set_defaults(spearman_brown=True)
    parser.add_argument("--save_null", dest="save_null", action="store_true")  # ttd: retain per-subject null in the .nc (for in-notebook SB recompute)
    parser.set_defaults(save_null=False)
    # MEG only: drop train images whose concept is in the test set (concept-level held-out, the THINGS-EEG2
    # design). Adds .exclude_test_concepts=1 to the load_dataset_kwargs cache segment.
    parser.add_argument("--exclude_test_concepts", dest="exclude_test_concepts", action="store_true")
    parser.set_defaults(exclude_test_concepts=False)

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
        ### normalization fix (Path 2): baseline z-score (Hebart native) instead of the
        ### per-(channel,time) total-variance z-score (zscore=from_raw). Adds
        ### .zscore=0.baseline_mode=zscore to the MEG cache segment => cold recompute.
        "zscore": 0,
        "baseline_mode": 'zscore',
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
    if args.rois == "none":
        # every sensor the MEG loader keeps. Pop, never assign: the MEG loader parses `rois` letter by
        # letter, so the literal "none" would select the occipital sensors.
        if args.dataset.startswith("things_eeg_2"):
            raise ValueError(
                "--rois none is MEG-only: the EEG loader defaults to rois='op', so dropping the key "
                "would load the occipital-parietal set. Use --rois all for every EEG channel."
            )
        load_dataset_kwargs.pop("rois", None)
    elif args.rois is not None:
        load_dataset_kwargs["rois"] = args.rois
    if args.exclude_test_concepts:
        if not args.dataset.startswith("things_meg"):
            raise ValueError("--exclude_test_concepts is MEG-only: THINGS-EEG2's train and test concepts are already disjoint.")
        load_dataset_kwargs["exclude_test_concepts"] = 1

    scorer_kwargs={
        ### fixed for now ###
        "model_name": args.model,
        #####################
    }
    
    match args.model:
        case "lda":
            scorer_kwargs["shrinkage"] = 1e-2
        case "linear":
            scorer_kwargs["l2_penalty"] = args.l2
        case "ridge":
            pass
        case _:
            pass

    if args.no_null:
        # skip the permutation null (and the null-derived n_decodable_dims); the scorer emits no
        # null key, so tt_decoding's null-gated branches skip accumulation + cluster-correction.
        scorer_kwargs["n_permutations"] = None

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
                "presentation_subset": args.sessions,
                "pca_source": args.pca_source,
                "per_image": args.per_image,
                "n_seeds": args.n_seeds,
                "spearman_brown": args.spearman_brown,
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
                "regress_out": args.regress_out,
                "fit_source": args.fit_source,
                "test_split_half": args.test_split_half,
                "save_null": args.save_null,
                "per_image": args.per_image,
                "n_seeds": args.n_seeds,
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
    



