# ThyroidNet: Training and Testing

This repository provides scripts to train and evaluate a thyroid nodule segmentation model using ultrasound images and their corresponding segmentation masks.

## Dataset structure

Both training and testing scripts expect a dataset root directory containing **`images/`** and **`masks/`** subfolders:

```text
dataset/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

The directory structure above is required by the scripts. Ensure each image has the corresponding ground-truth mask in the format expected by the dataset loader.

## Training

Provide the path to the training dataset using the required `--data` argument:

```bash
python train.py --data /path/to/train_dataset
```

The training script validates the dataset path and checks that both `images/` and `masks/` exist before calling `train(data_dir)`.

## Testing

Provide the path to the test dataset using the same `--data` argument:

```bash
python test.py --data /path/to/test_dataset
```

The testing script loads the model checkpoint from:

```text
best_model.pth
```

Place `best_model.pth` in the script's current working directory, or update `MODEL_FILE` in the testing script to point to your checkpoint. The script then calls `evaluate_test_set(TEST_IMAGES, TEST_MASKS, MODEL_FILE)`.

## Command-line argument

| Argument | Required | Description |
|---|---|---|
| `--data` | Yes | Root directory containing `images/` and `masks/` |

## Notes

- Replace 'best_model.pth' with the actual filename of the best model.