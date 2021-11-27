# FairCal: Fair Calibration for Face Verification

Code for the paper *FairCal: Fair Calibration for Face Verification*.

## Requirements

To install requirements:

```setup
conda install --file requirements.txt
```

The code assumes that the embeddings from the pre-trained models and the pairs and cosine similarities for both the RFW and BFW are contained in the folder data. Due to the licenses of the datasets, these cannot be shared. The following functions will need to be updated depending on how the embeddings and the pairs are saved:

- collect_embeddings_rfw, collect_embeddings_bfw, collect_embeddings_ijbc, collect_miscellania_rfw, collect_miscellania_bfw, collect_miscellania_ijbc in approaches.py
- collect_error_embeddings_rfw, collect_error_embeddings_bfw, collect_error_embeddings_ijbc in approaches_ftc
- lines 290-298 in main.py

The pre-trained models to generate the embeddings were obtained from the following repos:

- FaceNet (VGGFace2), FaceNet (Webface) - https://github.com/timesler/facenet-pytorch
- Arcface - https://github.com/onnx/models/tree/master/vision/body_analysis/arcface

These models have their own dependencies.

## Evaluating Methods

To run the experiments, run the following commands

```train
python main.py --dataset rfw --calibration_methods beta
python main.py --dataset bfw --calibration_methods beta
python main.py --dataset ijbc --calibration_methods beta
```

## Figures and Tables

Figures 1 and 2 in the paper were generated with the Jupyter Notebook "Main Figures", while the remaining figures and tables with the Jupiter Notebook "ICLR Images and Tables".
