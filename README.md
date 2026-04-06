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
  - [Running the demo](#running-the-demo)
  - [Reproducing the results](#reproducing-the-results)
- [Information about the datasets](#information-about-the-datasets)
  - [The THINGS EEG2 Dataset](#the-things-eeg2-dataset)
  - [The THINGS MEG Dataset](#the-things-meg-dataset)
- [License](#license)
- [References](#references)


# Overview

In this work, we characterize the multidimensional structure of stimulus-related signals in human EEG and MEG responses to natural images. We show that representational dimensionality rapidly expands after stimulus onset, peaking within 100 ms and gradually decaying over hundreds of milliseconds, and that these dynamics track decoding accuracy for both behavioral embeddings and deep neural network features. We further show that leading representational models do not fully explain neural dimensionality, and that the remaining unexplained variance carries behaviorally relevant information not captured by current models.


Here, we demonstrate how to:
 - [Install the code and libraries for analyses](#installation-guide;) 
 <!-- - [Compute scores and generate figures with a subset of data](#running-the-demo) -->
 - [Reproduce all results in the manuscipt](#reproducing-results)


# System requirements

## Hardware requirements
The code requires a standard computer with enough CPU and GPU compute power to support all operations. The scripts for replicating the main results use about ~24 GB GPU RAM at peak but also work with CPU only.


## Software requirements

### OS requirements
The code has been tested on RHEL 9.3.

### Python version
The code has been tested on Python==3.12.4.

### Python dependencies
The list of python libraries to run all scripts is in ```requirements.txt```.

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

<!-- ## Running the demo

In this demo, TODO -->

## Reproducing the results

Each dataset dataset has a corresponding ```dataset``` flag:
 - ```things_eeg_2```
 - ```things_meg```

Each set fo features of interest also has a corresponding ```analysis``` flag:
 - ```behaviorz```
 - ```model_srpz```
 - ```everythingz```
 - ```model_srpz_full```

They correspond to the behavioral embeddings, model feature maps from the first and last layers, the concatenation of the previous two, and model feature maps from all layers, which are for Supplement Figure 2 and 3. For the latter three, an addition```uid``` flag is needed---```openclip_rn50_yfcc15m```.

To compute results for Figure 2, run:
```
python scripts/compute_score.py --script ttpca --space eeg --dim1 target_var --dim2 neuroid --dim3 time --splitdim presentation --dataset {dataset}
python scripts/compute_score.py --script ttd --analysis {analysis} --model linear --dataset {dataset}
```
and run ```notebooks/figures/fig_2.ipynb``` .

To compute results for Figure 3, run:
```
python scripts/compute_score.py --script ttdgen --analysis behaviorz --model linear --dataset {dataset}
```
and run ```notebooks/figures/fig_3.ipynb``` .

To compute results for Figure 4, run:
```
python scripts/compute_score.py --script ttpca --space eeg --dim1 target_var --dim2 neuroid --dim3 time --splitdim presentation --dataset {dataset}
python scripts/compute_score.py --script ttd --analysis {analysis} --model plssvd --dataset {dataset}
```
and run ```notebooks/figures/fig_4.ipynb``` .

To compute results for Figure 5, run:
```
python scripts/compute_score.py --script rmv --dataset things_eeg_2  --pctime 0.2 --analysis behavior
python scripts/image_matching_experiment.py --num_subjects 50 --exp_id v00_time=02 --time .2
```
then upload the created files to web hosting provider like DreamHost, collect the data from online experiment website like Prolific, move the data to ```BONNER_CACHING_HOME / behavior / image_matching_experiment / v00_time=02 / data / responses```, and run ```notebooks/figures/fig_5d.ipynb``` . The results we collected were anonymized and are available at https://osf.io/y6znd and can be downloaded by running ```python scripts/download_behavior_data.py```.

To compute results for Supplement Figure 1, run:
```
python scripts/compute_score.py --script ttpca --space eeg --dim1 target_var --dim2 neuroid --dim3 time --splitdim presentation --dataset things_meg -rois f
```
and run ```notebooks/figures/sfig_1.ipynb``` .

To compute results for Supplement Figure 2, run:
```
python scripts/compute_score.py --script ttd --analysis model_srpz_full --uid openclip_rn50_yfcc15m --model linear --dataset {dataset}
```
and run ```notebooks/figures/sfig_2.ipynb``` .

To compute results for Supplement Figure 4, run:
```
python scripts/compute_score.py --script ttd --analysis model_srpz_full --uid openclip_rn50_yfcc15m --model plssvd --dataset {dataset}
```
and run ```notebooks/figures/sfig_4.ipynb``` .

To compute results for Supplement Figure 5, run:
```
python scripts/image_matching_experiment.py --num_subjects 50 --exp_id v01_time=01_pct=65 --time .1
```
then repeat the procedure as for Figure 5 and run ```notebooks/figures/sfig_5.ipynb```. The results we collected were anonymized and are available at https://osf.io/y6znd and can be downloaded by running ```python scripts/download_behavior_data.py```.

# Information about the datasets

## The THINGS EEG2 Dataset

THINGS EEG2 [1] is a large-scale EEG dataset containing brain responses from 10 human subjects to images of everyday objects drawn from the THINGS database [2]. Stimuli were presented for 100 ms followed by 100 ms fixation. The training set includes four repetitions of responses to 10 unique images per 1,654 object concepts; the test set includes 80 repetitions of responses to one image per 200 concepts (non-overlapping with training concepts). Epochs span −200 ms to 800 ms relative to stimulus onset. Analyses used 17 channels over occipital and parietal regions as preprocessed by the original authors.

## The THINGS MEG Dataset

THINGS MEG [3] is a large-scale MEG dataset containing brain responses from 4 human subjects to images from the THINGS database [2]. Stimuli were presented for 500 ms followed by 1 s fixation (±200 ms jitter). The training set includes 12 unique images per 1,854 concepts, each presented once; the test set includes one image per 200 concepts, each presented 12 times. Epochs span −100 ms to 1300 ms relative to stimulus onset. Analyses used 80 channels over occipital and parietal regions.

# License

This project is covered under the MIT License.

# References

[1] Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. *NeuroImage*, 264, 119754.

[2] Hebart, M. N., Dickter, A. H., Kidder, A., Kwok, W. Y., Corriveau, A., Van Wicklin, C., & Baker, C. I. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PLoS ONE*, 14(10), e0223792.

[3] Hebart, M. N., Contier, O., Teichmann, L., Rockter, A. H., Zheng, C. Y., Kidder, A., Corriveau, A., Vaziri-Pashkam, M., & Baker, C. I. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. *eLife*, 12, e82580.

