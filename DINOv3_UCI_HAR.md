# CoTMix + DINOv3 on UCI-HAR

## What changes

- `CNN` remains available as the original CoTMix baseline.
- `DINOv3ViTS16` uses DINOv3 ViT-S/16 as **backbone = encoder**.
- The original CoTMix temporal mixup is still performed on `[B, C, T]` time-series tensors.
- Inside the DINOv3 encoder, each mixed or original time-series window is converted to a 3-channel Gramian Angular Field (GAF) image, resized to 224x224, and passed to DINOv3. The CLS embedding (normally 384 dimensions) is sent to the classifier.
- The GAF-to-RGB grouping is an experimental adapter: for UCI-HAR's nine channels, the three groups of three channels are averaged into the three image planes. This is not part of the original CoTMix paper and should be treated as an ablation/design choice.
- DINOv3 is frozen by default. Pass `--unfreeze_dinov3` to fine-tune it.

## Install

Run from the repository root:

```bash
pip install -r requirements-dinov3.txt
```

Install the other dependencies listed/required by the original CoTMix repository as usual. Loading pretrained DINOv3 downloads the model from Hugging Face and may require accepting its model access terms.

## Expected UCI-HAR data layout

The existing CoTMix loader expects preprocessed PyTorch files, not the raw UCI-HAR text files:

```text
data/
└── HAR/
    ├── train_2.pt
    ├── test_2.pt
    ├── train_11.pt
    ├── test_11.pt
    └── ...
```

Each `.pt` file must load to a dictionary with `samples` and `labels`, matching the original repository's `dataloader.py`. `samples` should represent windows with nine sensor channels and 128 time steps; the loader converts the tensor to `[N, C, T]` when needed. Subject IDs must match `configs/data_model_configs.py`'s HAR `scenarios` list. The ZIP does not contain the actual preprocessed UCI-HAR `.pt` data files, so those need to be prepared separately.

## First smoke run

Start with one scenario/run and a small epoch count:

```bash
python main.py --dataset HAR --backbone DINOv3ViTS16 --data_path ./data/ --device cuda:0 --num_runs 1 --batch_size 16 --epochs 2
```

For CPU-only testing:

```bash
python main.py --dataset HAR --backbone DINOv3ViTS16 --data_path ./data/ --device cpu --num_runs 1 --batch_size 2 --epochs 1
```

The CPU command may be very slow because DINOv3 image encoding is substantially heavier than the original 1D-CNN.

## Fine-tuning

```bash
python main.py --dataset HAR --backbone DINOv3ViTS16 --data_path ./data/ --device cuda:0 --num_runs 1 --batch_size 8 --epochs 20 --unfreeze_dinov3 --dinov3_lr 0.00001 --classifier_lr 0.001
```

## Important interpretation note

This is an experimental adaptation of a vision encoder to sensor time series, not a drop-in replacement with equivalent input semantics. Compare it against the original CNN with the same source/target subject splits and random seeds. For a fair comparison, record the GAF adapter, pretrained checkpoint, frozen/fine-tuned setting, and exact package versions.
