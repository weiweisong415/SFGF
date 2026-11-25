# SFGF
This is the code implementation for our paper "SFGF: Spatial-Frequency Guidance Fusion for Joint Classification of Hyperspectral and LiDAR Data"

SFGF is a dual-branch network that jointly classifies hyperspectral (HSI) and LiDAR data. The two modality branches exchange information through **SFGF blocks**: three cross-modal adaptive filter generators — **ASFG** (spatial-semantic, `spatial.py`), **ALFG** (low-pass) and **AHFG** (high-pass) (`model.py`) — generate filters from one modality that dynamically re-calibrate the features of the other, realizing bidirectional guided filtering in both the spatial and frequency domains at multiple scales.

## Code structure

```
main.py        # training / evaluation entry
funcs.py       # SFGF network, SFGFBlock, train and evaluation
model.py       # frequency-domain guided filtering: ALFG + AHFG
spatial.py     # spatial-domain guided filtering: ASFG
```

## Requirements

- Python >= 3.8, PyTorch (CUDA)
- See `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data preparation

Place the datasets under `./dataset` as follows (each `*_TR_20_{i}.mat` contains the training map `TRLabel` and test map `TSLabel` for the *i*-th run, with 20 training samples per class):

```
dataset/
├── MUUFL/
│   ├── MUUFL.mat                  # keys: HSI, LiDAR
│   └── MUUFL_TR_20_{1..10}.mat
├── Houston2013/
│   ├── Houston2013_HSI.mat        # key: HSI
│   ├── Houston2013_DSM.mat        # key: DSM
│   └── Houston2013_TR_20_{1..10}.mat
└── Trento/
    ├── HSI.mat                    # key: HSI
    ├── LiDAR.mat                  # key: LiDAR
    └── Trento_TR_20_{1..10}.mat
```

## Usage

The three datasets are selected with `--dataset` (default: `muufl`). Dataset-specific hyper-parameters (patch size, epochs, loss weights, number of runs) are applied automatically; any of them can be overridden on the command line.

```bash
# MUUFL (default)
python main.py
# Houston2013
python main.py --dataset houston
# Trento
python main.py --dataset trento

# example of overriding hyper-parameters
python main.py --dataset muufl --runs 10 --epoch 600 --apha 0.01 --gamma 0.01 --lamda 1.0
```

## Outputs

Results are written to `results/<dataset>/`:

- `net_params_<dataset>.pkl` — best model weights
- `metrics.json` — OA / AA / Kappa of every run

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{SFGF,
  title={SFGF: Spatial-frequency Guidance Fusion for Joint Classification of Hyperspectral and LiDAR Data},
  author={},
  journal={},
  year={}
}
```
