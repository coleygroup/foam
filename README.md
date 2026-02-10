# FOAM: Formula-constrained Optimization for Annotating Metabolites

This repository complements the paper, "Generative structural elucidation from mass spectra as a iterative optimization problem". 

FOAM is a method for _de novo_ structural elucidation from tandem mass spectra that frames the problem as iterative optimization. Given an experimental spectrum and molecular formula, FOAM uses a formula-constrained graph genetic algorithm to propose and refine candidate structures, scoring them with ICEBERG (a learned spectral simulator) to maximize predicted spectral similarity to the observed spectrum. As a spectrum-conditioned generative method that can also accept any number of seed structures, FOAM supports the refinement of structures from any sourcs---virtual libraries, generative models, or other domain-specific tools.


## System Requirements

### Software and operating system dependencies

**Python:** >= 3.9, <3.12 (tested with 3.10). 
**Operating System:** 
* Our experiments were run on Linux (x86_64) systems with both CPU-only, GPU-accelerated, and multi-GPU configurations.
* Other operating systems, such as macOS, have been tested only preliminarily (e.g. installation) but may not have been fully validated for large-scale experiments.

### Hardware Requirements

- **CPU-only mode:** No special hardware required. Any modern x86_64 or ARM64 processor is sufficient.
- **GPU-accelerated mode:** NVIDIA GPU with CUDA 12.1 support (Linux only). The number of GPU workers is configurable via `--gpu-workers`.
- **RAM:** 20 GB minimum recommended; large seed cache datasets and/or extended population sizes may require additional RAM.
- **Disk:** ~5 GB for dependencies; additional space (~20 GB) depending on downloaded data files and model checkpoints.

## Overview

**Repository Structure**
```
foam/
├── configs/              # YAML configuration files for experiments
├── data/                 # Data assets and downloads (please download separately; see Data section)
├── notebooks/            # Jupyter notebooks for analysis and examples
├── run_scripts/          # Shell scripts for running experiments
├── src/foam/             # Main package source code
│   ├── opt_graph_ga_fc/  # Formula-constrained GraphGA optimizer
│   ├── utils/            # Utility modules (chemical, parallel, plotting)
│   ├── base_opt.py       # Abstract optimizer base class
│   ├── evaluators.py     # Evaluation metrics and logging
│   └── oracles.py        # Scoring functions (spectral similarity, etc.)
└── testing/              # Test scripts and fixtures

```



**Tasks:**
-  **Simulated spectrum matching:** A common workflow is to build models that fragment molecules _in silico_. Then, a database of molecules can be fragmented to find the molecule with the most similar putative spectrum to the unknown spectrum of interest. This same type of workflow for predicting forward a spectrum from a molecule and matching that to an unknown empirical spectrum can be used with NMR and other types of data. To recapitulate this setting, we train a "forward neural network" that predicts spectra from molecules. We then use this model to fragment 200 molecules, which form the basis of this structure rediscovery task. In the optimization loop, the forward simulator is run on each generated molecule and its nearness to the unknown molecule is calculated.



## Installation

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or conda + pip
- Git

### Option 1: Using mamba + pip

```bash
# 1. Set up the ms-pred environment
cd ms-pred
mamba env create -f environment.yml
mamba activate ms-gen
pip install -r requirements.txt

# 2. Install FOAM
cd ../foam
pip install -r requirements.txt
python setup.py develop
```

### Option 2: Using uv (faster, recommended!)
These use updated dependencies. 
```bash
# 1. Create and activate a virtual environment
git clone https://github.com/coleygroup/foam.git
cd foam
uv venv --python 3.10 ms-gen-foam
source ms-gen-foam/bin/activate

# 2. Install ms-pred
cd .. && git clone https://github.com/coleygroup/ms-pred.git && cd ms-pred
# for up to date installation: use updated dependencies on the branch fix-deps.
git checkout fix-deps
uv pip install -e .

# 3. Install FOAM
cd ../foam

# For CPU-only:
uv pip install -e ".[cpu]"

# For GPU (Linux with CUDA 12.1):
uv pip install -e ".[cuda]"
```


### Typical Install Time

Installation takes approximately **10-20 minutes** on a normal desktop computer with a broadband internet connection. The majority of this time is spent downloading PyTorch and its dependencies. If you wish to accelerate this installation, consider running installation with `uv` instead, which will speed up downloads and dependency resolution to ~1 minute. 

## Downloading models and data

### Models
ICEBERG (cite) is used for forward spectral simulation; given a SMILES representation of a molecule alongside instrumental metadata such as specified adducts, collision energy, and instrument type, a spectrum can be generated.

Model weights: 
* MassSpecGym: weights available here: https://zenodo.org/uploads/18502041
* NIST'20: since this model is trained on a commercial library, we cannot make these weights publicly available. Please email mrunali@mit.edu if you already have a NIST license and wish to obtain the weights.

### Data
We primarily evaluate FOAM on a structure rediscovery task using spectra from the NIST 2020 (NIST20) reference library, as well as the open-access MassSpecGym benchmark. For ease of querying, test spectra are provided to FOAM via compressed hdf5 dictionaries that can be indexed by an ID. Since the NIST20 reference library is not open-access, we cannot provide these spectra publically. Code to package the MassSpecGym spectra into a HDF5 file can be found in the ms-pred repository. 

FOAM initializes its search with a set of formula-matched structures. For the NIST20 evaluation, we retrieve all seeds from PubChem, excluding the true structure (this exclusion is applied after all 2D structures are loaded for a requested formula in our code, and only applied if the true SMILES is provided). Structure caches for both PubChem and DiffMS are provided in our Zenodo and should be deposited into the `data/{pubchem}|{diffms}/` directories. 


If a better or different set of seeds is known, they may also be supplied as a .txt file of SMILES to the keyword argument `seed-lib-dir`. To use an additional set of seeds beyond those in the PubChem dataset, pass them as a HDF5 file indexed by either formula or by the same ID as the test spectrum with `--extra-seeds` [beta, supporting additional file types soon]. 


## Demo

### Running the Demo

After completing installation and downloading the data and model files, you can run a small demo to verify everything works:

```bash
python src/foam/opt_graph_ga_fc/run_opt.py \
    --seed 42 \
    --num-workers 4 \
    --device cpu \
    --batch-size 16 \
    --wandb disable \
    --gen-model-ckpt <path-to-iceberg-gen-checkpoint> \
    --inten-model-ckpt <path-to-iceberg-inten-checkpoint> \
    --ignore-precursor-peak \
    --max-nodes 100 \
    --oracle-type Cos_SA_ \
    --criteria entropy \
    --multiobj \
    --eval-names NDSBestMol TopNDSScore NDSParetoRanking NDSFronts InchiKeyMatch \
    --spec-id nist_1337913 \
    --spec-lib-dir data/spec_datasets/nist23/spec_files.hdf5 \
    --spec-lib-label data/spec_datasets/nist23/labels.tsv \
    --seed-lib-dir data/pubchem/pubchem_formulae_inchikey.hdf5 \
    --max-seed-sim 0.95 \
    --top-k 1 5 10 \
    --max-calls 200 \
    --keep-population 200 \
    --population-size 50 \
    --offspring-size 100 \
    --num-islands 1 \
    --threshold 0.0 \
    --starting-seed-size 100 \
    --selection-sorting-type cand_crowding \
    --parent-tiebreak cand_crowding \
    --truncate \
    --save-dir results/demo
```

Alternatively, use a config file:
```bash
python src/foam/opt_graph_ga_fc/run_opt.py --config configs/final/202506_multiobjective_nist_10k_calls.yaml
```

### Expected Output

The optimization run will produce a directory under `--save-dir` (e.g., `results/demo/`) containing:
- Optimization logs and metrics (e.g., best scores per generation)
- Candidate molecules generated during the run
- Evaluation metrics (NDS scores, Pareto rankings, InChIKey matches)
- Wandb logs (if enabled)

### Expected Runtime

<!-- TODO: verify these numbers -->
| Configuration | Hardware | Approximate Runtime |
|---|---|---|
| Demo (200 calls, CPU) | Standard desktop (4 cores) | ~4 minutes |
|1,000 calls, CPU only | Standard desktop (8 cores) | ~20 minutes |
|1,000 calls, GPU-enabled | NVIDIA H100 | ~4 minutes |
|7,500 calls, GPU-enabled | NVIDIA H100 | ~20 minutes |

## Instructions for Use

### Running FOAM on your data

FOAM can be run on the command line with arguments supplied to `src/foam/opt_graph_ga_fc/run_opt.py`, or with a YAML config file using the `--config` flag. Example configs are in `configs/final/`.

To run on your own spectrum of interest:
1. Prepare your spectrum in HDF5 format (matching the structure of the provided `spec_files.hdf5`), or provide a folder of spectra.
2. Optionally prepare a seed library (`.hdf5` or `.txt` file of SMILES).
3. Run the optimizer, pointing to your data with `--spec-lib-dir`, `--spec-lib-label`, and `--spec-id`.

### Description of Parameters
```
    # System/overall parameters
    --seed {int}              # numerical seed to set
    --num-workers {int}       # number of cpu workers
    --device {str}            # whether to use cpu or gpu for ICEBERG inference
    --gpu-workers {int}       # num. of ICEBERG copies on GPU
    --batch-size {int}        # if doing parallel inference: batch size (trade off with `gpu_workers`)
    --wandb {str}             # options for wandb logging (disable, offline, online)
    # ICEBERG-specific parameters
    --gen-model-ckpt {path}   # path to iceberg-generate model ckpt
    --inten-model-ckpt {path} # path to iceberg-score model ckpt
    --ignore-precursor-peak   # if included, will exclude precursor peak from scoring
    --max-nodes {int}         # num. nodes for ICEBERG to predict
    # Scoring parameters
    --oracle-type {str}       # type of oracle to use
    --criteria {str}          # scoring criteria (cosine, entropy)
    --multiobj                # if included, will use multiobj setup
    --eval-names {str}        # list of evaluation metrics to track
    --top-k {num}             # list of k values at which to compute metrics
    # Spectrum and seeding parameters
    --spec-id {str}           # if provided, spectrum name to look up in `spec-lib-dir`
    --spec-lib-dir {str}      # path to spectra (can be .hdf5 or folder)
    --spec-lib-label {str}    # metadata for spectra
    --seed-lib-dir {str}      # path to seed structures (folder or .txt)
    --max-seed-sim 0.95       # for evaluation only: maximum Tanimoto seed similarity to target mol
    # Population parameters
    --max-calls {int}         # num of max calls to ICEBERG
    --keep-population {int}   # num of candidates to keep in buffer as history (ideally, set to max-calls + anticipated seed set size)
    --population-size {int}   # num of candidates to keep in the population (bottleneck & mating pool size)
    --offspring-size {int}    # num of offspring to generate
    --num-islands {int}       # num of islands to run in parallel
    --threshold {int}         # score threshold
    --starting-seed-size {int}# num of seeds to use
    # Advanced parameters
    --selection-sorting-type  # sort by tanimoto crowding or by objective crowding (default: cand_crowding)
    --parent-tiebreak         # how to break ties when choosing parents (default: cand_crowding)
    --truncate                # remove molecules with low entropy similarities
    --use-clustered-evs       # use only representative collision energies
    --mutate-parents          # dedicate some generative burden to mutating parents directly (not recommended)
    --use-iceberg-spectra     # toy task: use ICEBERG-simulated spectra as ground-truth
    --save-dir {path}         # directory to save results
    --tags {str ...}          # tags for experiment tracking
```

<!-- ### Variants of FOAM
* **Single-objective:** do not pass `--multiobj`
* **Multi-objective:** pass `--multiobj`
* **Use ICEBERG spectra as ground truth:** pass `--use-iceberg-spectra`
* **Separated collision energies:** change `--oracle-type Cos_SA_ColliEng_` -->

### Reproduction Instructions

To reproduce the results from the paper, use the config files provided in `configs/final/`. For example:

```bash
# Multi-objective optimization on NIST with 7,500 calls
python src/foam/opt_graph_ga_fc/run_opt.py --config configs/final/nist_7.5k_calls.yaml

# Multi-objective on MassSpecGym with pubchem and diffms seeds
python src/foam/opt_graph_ga_fc/run_opt.py --config configs/final/msg_pubchem_diffms_1500calls.yaml
```

Note: Config files contain placeholder paths to model checkpoints and data files. You will need to update these paths to match your local setup before running.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

