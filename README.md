# Contents

- [Contents](#contents)
- [Overview](#overview)
- [System requirements](#system-requirements)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
    - [OS requirements](#os-requirements)
    - [Python version](#python-version)
    - [Python dependencies](#python-dependencies)
- [Installation guide](#installation-guide)
  - [Reproducing the results](#reproducing-the-results)
- [Information about the datasets](#information-about-the-datasets)
  - [The THINGS EEG2 Dataset](#the-things-eeg2-dataset)
  - [The THINGS MEG Dataset](#the-things-meg-dataset)
- [License](#license)
- [References](#references)


# Overview

In this work, we characterize the multidimensional structure of stimulus-related signals in human EEG and MEG responses to natural images. We show that representational dimensionality rapidly expands after stimulus onset, peaking within 100 ms and gradually decaying over hundreds of milliseconds, and that these dynamics track decoding accuracy for both behavioral embeddings and deep neural network features. We further show that leading representational models do not fully explain neural dimensionality, and that the remaining unexplained variance carries behaviorally relevant information not captured by current models.


Here, we demonstrate how to:
 - [Install the code and libraries for analyses](#installation-guide)
 - [Reproduce all results in the manuscript](#reproducing-the-results)


# System requirements

## Hardware requirements
The code requires a standard computer with enough CPU and GPU compute power to support all operations. The scripts for replicating the main results use about ~24 GB GPU RAM at peak but also work with CPU only, except the permutation nulls that ```notebooks/figures/sfig_5.ipynb``` computes on its first run, which need a CUDA GPU.


## Software requirements

### OS requirements
The code has been tested on RHEL 9.3.

### Python version
The code has been tested on Python==3.12.4.

### Python dependencies
The list of python libraries to run all scripts is in ```requirements.txt```.

```deepjuice```, which extracts the DNN feature maps, is a restricted dependency available from its author, so it is not in ```requirements.txt```. With access to its repository, install it with ```pip install -r requirements_deepjuice.txt```. It is needed only by the ```compute_score.py``` commands that take a ```--uid``` flag and by ```notebooks/figures/sfig_5.ipynb```; the code imports without it, and constructing a ```DJModel``` without it raises an ImportError.

# Installation guide

Clone this repository and navigate to the repository folder.
```
git clone https://github.com/zche377/temporal_dimensionality.git
cd temporal_dimensionality
```

Copy ```.env.example``` to ```.env``` and set the paths for where the data, models, intermediate and final results are saved.
```
cp .env.example .env
```

Install required packages and activate the environment. (~3 minutes)
```
conda env create -f environment.yml
conda activate tdim
```

## Reproducing the results

Run every command from the repository root. Each ```compute_score.py``` command writes its results under ```BONNER_CACHING_HOME```; each notebook in ```notebooks/figures/``` reads them and saves its figure under ```artifacts/figures/```. ```{dataset}``` is ```things_eeg_2``` or ```things_meg```. For ```things_meg```, add ```--exclude_test_concepts``` to every ```compute_score.py``` command: it drops from the MEG training set every image whose concept is in the MEG test set, as in the design of THINGS EEG2. The commands with ```--uid openclip_rn50_yfcc15m``` extract DNN features and need ```deepjuice``` (see [Python dependencies](#python-dependencies)).

The cvPCA neural dimensionality used by Figures 2-4 and Figure S3 (Stringer per-image split-half, 10 seeds, no Spearman-Brown correction):
```
python scripts/compute_score.py --script ttpca --space eeg --dim1 target_var --dim2 neuroid --dim3 time --splitdim presentation --per_image --n_seeds 10 --no_sb --dataset {dataset}
```

### Figure 1
A schematic; no code.

### Figure 2
Run the cvPCA command above, then
```
python scripts/compute_score.py --script ttd --analysis behaviorz --model linear --dataset {dataset}
python scripts/compute_score.py --script ttd --analysis model_srpz --uid openclip_rn50_yfcc15m --model linear --dataset {dataset}
```
and run ```notebooks/figures/fig_2.ipynb```.

### Figure 3
Run the cvPCA command above, then
```
python scripts/compute_score.py --script ttdgen --analysis behaviorz --model linear --dataset {dataset}
```
and run ```notebooks/figures/fig_3.ipynb```.

### Figure 4
Run the cvPCA command above, then
```
python scripts/compute_score.py --script ttd --analysis behaviorz --model plssvd --per_image --n_seeds 10 --dataset {dataset}
python scripts/compute_score.py --script ttd --analysis model_srpz --uid openclip_rn50_yfcc15m --model plssvd --per_image --n_seeds 10 --dataset {dataset}
python scripts/compute_score.py --script ttd --analysis everythingz_all_layer --uid openclip_rn50_yfcc15m --model plssvd --per_image --n_seeds 10 --dataset {dataset}
```
and run ```notebooks/figures/fig_4.ipynb```.

### Figure 5
Panel a comes from
```
python scripts/compute_score.py --script rmv --dataset things_eeg_2 --pctime 0.2 --analysis behavior
```
which writes it to ```BONNER_CACHING_HOME/main_analyses/remaining_var_pdistance.pca=False.test_images=True/regress_out=behavior/dataset=things_eeg_2/load_dataset_kwargs=default/scorer_kwargs=model_name=linear.l2_penalty=0.01/target_timepoint=0.2.png```. Panel b is a schematic. For panel c, download the anonymized responses of both image-matching experiments (https://doi.org/10.5281/zenodo.21937065) into ```BONNER_CACHING_HOME```:
```
python scripts/download_behavior_data.py
```
and run ```notebooks/figures/fig_5c.ipynb```. To run the experiments yourself, create them with
```
python scripts/image_matching_experiment.py --num_subjects 50 --exp_id v00_time=02 --time .2
python scripts/image_matching_experiment.py --num_subjects 50 --exp_id v01_time=01_pct=65 --time .1
```
upload the created files to a web host, collect responses through an online platform such as Prolific, and place each experiment's response files in ```BONNER_CACHING_HOME/behavior/image_matching_experiment/{exp_id}/data/responses```.

### Figure S1
```
python scripts/compute_score.py --script ttpca --space eeg --dim1 target_var --dim2 neuroid --dim3 time --splitdim presentation --per_image --n_seeds 10 --no_sb --rois f --dataset {dataset}
```
then run ```notebooks/figures/sfig_1.ipynb```.

### Figure S2
```
python scripts/compute_score.py --script ttd --analysis model_srpz_full --uid openclip_rn50_yfcc15m --model linear --dataset {dataset}
```
then run ```notebooks/figures/sfig_2.ipynb```.

### Figure S3
Run the cvPCA command above for ```things_eeg_2```, then run ```notebooks/figures/sfig_3.ipynb```.

### Figure S4
```
python scripts/compute_score.py --script ttd --analysis model_srpz_full --uid openclip_rn50_yfcc15m --model plssvd --per_image --n_seeds 10 --dataset {dataset}
```
then run ```notebooks/figures/sfig_4.ipynb```.

### Figure S5
Run Figure 2's ```model_srpz``` command for ```things_eeg_2``` (the notebook checks its EEG null computation against those scores), then
```
python scripts/compute_score.py --script ttd --analysis model_srpz --uid openclip_rn50_yfcc15m --model linear --regress_out behavior --no_null --dataset {dataset}
```
and run ```notebooks/figures/sfig_5.ipynb```. On its first run the notebook computes the permutation nulls and caches them under ```BONNER_CACHING_HOME/main_analyses/```; this needs ```deepjuice``` and a CUDA GPU.

# Information about the datasets

## The THINGS EEG2 Dataset

THINGS EEG2 [1] is a large-scale EEG dataset containing brain responses from 10 human subjects to images of everyday objects drawn from the THINGS database [2]. Stimuli were presented for 100 ms followed by 100 ms fixation. The training set includes four repetitions of responses to 10 unique images per 1,654 object concepts; the test set includes 80 repetitions of responses to one image per 200 concepts (non-overlapping with training concepts). Epochs span −200 ms to 800 ms relative to stimulus onset. Analyses used 17 channels over occipital and parietal regions as preprocessed by the original authors.

## The THINGS MEG Dataset

THINGS MEG [3] is a large-scale MEG dataset containing brain responses from 4 human subjects to images from the THINGS database [2]. Stimuli were presented for 500 ms followed by 1 s fixation (±200 ms jitter). The training set includes 12 unique images per 1,854 concepts, each presented once; the test set includes one image per 200 concepts, each presented 12 times. Epochs span −100 ms to 1300 ms relative to stimulus onset. Analyses used 80 channels over occipital and parietal regions.

# License

This project is covered under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full text.

# References

[1] Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. *NeuroImage*, 264, 119754.

[2] Hebart, M. N., Dickter, A. H., Kidder, A., Kwok, W. Y., Corriveau, A., Van Wicklin, C., & Baker, C. I. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PLoS ONE*, 14(10), e0223792.

[3] Hebart, M. N., Contier, O., Teichmann, L., Rockter, A. H., Zheng, C. Y., Kidder, A., Corriveau, A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. *eLife*, 12, e82580.

