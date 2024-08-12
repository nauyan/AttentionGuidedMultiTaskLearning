# Attention Guided Multi-Task Learning

This is an offical implemetation of the paper titled "Attention-Guided Multi-Task Learning for Enhanced Cancer Grading and Survival Prediction". The workflow is organized into four main scripts that should be executed in sequence.

## Installation

To set up the environment, use the provided `env.yml` file. This will install all the necessary dependencies.

``` bash
conda env create -f env.yaml
conda activate your_environment_name
```

## Configuration File

Before running any scripts, ensure that a configuration file is present in the configuration directory. The file should contain settings for the dataset, model, dataloader, losses, evaluators, and training options.

```json
{
    "dataset_metadata": {
        "train_images": "path_to_train_images",
        "test_images": "path_to_test_images",
        "train_metadata": "path_to_train_metadata",
        "test_metadata": "path_to_test_metadata",
        "num_classes": 4,
        "task": "classification",
        "model_name": "conv_trans",
        "loss_name": "cox",
        "optimizer_name": "adam",
        "learning_rate": 0.01,
        "checkpoints_name": "model_checkpoints"
    },
    "model_configs": {
        "model_name": "conv_trans",
        "cnn_encoder": "encoder_type",
        "cnn_encoder_pretrained": false,
        "vit_arch": "vit_architecture",
        "vit_arch_pretrained": true,
        "nc_net_encoder": false
    },
    "dataloader_configs": {
        "train_dataloader": {
            "batch_size": 20
        },
        "test_dataloader": {
            "batch_size": 20
        }
    },
    "losses": {
        "cross_entropy": {}
    },
    "evaluators": {
        "accuracy": {},
        "roc_auc_score": {}
    },
    "tensorboard_logging": true,
    "epochs": 500
}
```

## Key Descriptions

### dataset_metadata
Contains information about the dataset used for training and testing.

- **train_images**: Path to the training images in .npy format.
- **test_images**: Path to the testing images in .npy format.
- **train_metadata**: Path to the CSV file containing metadata for the training dataset.
- **test_metadata**: Path to the CSV file containing metadata for the testing dataset.
- **num_classes**: Number of classes for the classification task.
- **task**: Specifies the type of task, e.g., "classification".
- **model_name**: Name of the model architecture to be used.
- **loss_name**: Name of the loss function.
- **optimizer_name**: Name of the optimizer.
- **learning_rate**: Learning rate for training.
- **checkpoints_name**: Name for the model checkpoints.

### model_configs
Configuration settings for the model.

- **model_name**: Name of the model architecture.
- **cnn_encoder**: Encoder type for CNN.
- **cnn_encoder_pretrained**: Boolean indicating whether to use a pretrained CNN encoder.
- **vit_arch**: Vision Transformer architecture.
- **vit_arch_pretrained**: Boolean indicating whether to use a pretrained Vision Transformer.
- **nc_net_encoder**: Boolean indicating whether to use a non-convolutional encoder.

### dataloader_configs
Configurations for data loading during training and testing.

- **train_dataloader**: Settings for the training data loader.
  - **batch_size**: Batch size for training.
- **test_dataloader**: Settings for the testing data loader.
  - **batch_size**: Batch size for testing.

###


## Workflow

### 1.Prepare Dataset
Before we start preparing the data we need to download the dataset for TCGA-GBMLGG and TCGA-KIRC. The dataset can be downloaded using the Google Drive [link](https://drive.google.com/drive/folders/14TwYYsBeAnJ8ljkvU5YbIHHvFPltUVDr).

```bash
python prepare_dataset.py
```
This script prepares the dataset by creating training and testing metadata CSV files.

#### Variables
`dataset_name`: specifies the name of the dataset

### 3.Extract Patches

This script extracts 4k patches from the Whole-Slide-Image

```bash
python extract_patches_4k.py
```
#### Variables
`dataset_name`: specifies the name of the dataset
`PARALLEL_RUN`: flag to enable or disable parallel processing

### 3.Extract Features

This script extracts features from the 4k patches

```bash
python extract_features_4k.py
```

#### Variables

`dataset_name` Name of the dataset
`sz` Size of the patches
`step` Step size for a given batch

### 4.Train Model

This is the training script to perform training on the extracted features

```bash
python train.py
```

#### Variables
`epochs:` Number of training epochs
`tensorboard_logging:` Flag to enable or disable TensorBoard logging.
`optimizer:`  Optimizer to use
`lr:` learning rate for the optimizer
