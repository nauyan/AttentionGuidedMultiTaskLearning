import json
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split


def get_brca_dataset(configs):

    # features4k_dir = "dataset/features_4k"
    features4k_paths = glob(f"{configs['features_dir']}/*")

    # dataset_metadata_path = "dataset/labels/brca.csv"
    dataset_metadata = pd.read_csv(configs['metadata'])

    metadata = pd.DataFrame(columns=['file_path', "classification_label"])
    tcga_ids = []
    for features4k_path in features4k_paths:
        feature4k_id = os.path.basename(features4k_path).replace(
            ".pt", "").split("_")[0]
        tcga_ids.append(feature4k_id)
        label = dataset_metadata[dataset_metadata['ID'] ==
                                 feature4k_id]['label']
        if label.shape[0] == 0:
            continue
        df = pd.DataFrame({
            "file_path": features4k_path,
            "classification_label": label
        })
        metadata = pd.concat([metadata, df], axis=0)

    metadata['classification_label'].replace({
        'ductal': 0,
        'lobular': 1
    },
                                             inplace=True)
    metadata_train, metadata_test = train_test_split(metadata, test_size=0.2)
    metadata_train.reset_index(inplace=True)
    metadata_test.reset_index(inplace=True)

    metadata_train.to_csv(configs['train_metadata'], index=False)
    metadata_test.to_csv(configs['test_metadata'], index=False)


def get_gbm_lgg_dataset_1024(configs):

    grade_metadata = pd.read_csv(f"{configs['metadata_dir']}/grade_data.csv")
    grade_metadata = grade_metadata.dropna(subset=['Grade'])
    grade_metadata['Grade'].replace({
        2.0: 0,
        3.0: 1,
        4.0: 2,
    }, inplace=True)

    survial_metadata = pd.read_csv(
        f"{configs['metadata_dir']}/all_dataset.csv")

    metadata = pd.DataFrame(columns=['file_path', "classification_label"])
    for idx in range(grade_metadata.shape[0]):
        image_path = glob(
            f"{configs['patches_dir']}/{grade_metadata['TCGA ID'][idx]}*")[0]

        df = pd.DataFrame({
            "file_path": [image_path],
            "classification_label": [grade_metadata["Grade"][idx]],
            "survival_time":
            survial_metadata[survial_metadata["TCGA ID"] ==
                             grade_metadata["TCGA ID"][idx]]
            ["Survival months"],
            "censored":
            survial_metadata[survial_metadata["TCGA ID"] ==
                             grade_metadata["TCGA ID"][idx]]["censored"]
        })
        metadata = pd.concat([metadata, df], axis=0)

    metadata_train, metadata_test = train_test_split(metadata, test_size=0.2)
    metadata_train.reset_index(inplace=True)
    metadata_test.reset_index(inplace=True)

    print(f"Saving Train CSV with {metadata_train.shape[0]} Samples")
    print(f"Saving Test CSV with {metadata_test.shape[0]} Samples")
    metadata_train.to_csv(configs['train_metadata'], index=False)
    metadata_test.to_csv(configs['test_metadata'], index=False)


def get_gbm_lgg_dataset_512(configs):

    grade_metadata = pd.read_csv(f"{configs['metadata_dir']}/grade_data.csv")
    grade_metadata = grade_metadata.dropna(subset=['Grade'])
    grade_metadata['Grade'].replace({
        2.0: 0,
        3.0: 1,
        4.0: 2,
    }, inplace=True)

    survial_metadata = pd.read_csv(
        f"{configs['metadata_dir']}/all_dataset.csv")

    metadata = pd.DataFrame(columns=['file_path', "classification_label"])
    for idx in range(grade_metadata.shape[0]):
        image_paths = glob(
            f"{configs['patches_dir']}/{grade_metadata['TCGA ID'][idx]}*")
        for image_path in image_paths:
            df = pd.DataFrame({
                "file_path": [image_path],
                "classification_label": [grade_metadata["Grade"][idx]],
                "survival_time":
                survial_metadata[survial_metadata["TCGA ID"] ==
                                 grade_metadata["TCGA ID"][idx]]
                ["Survival months"],
                "censored":
                survial_metadata[survial_metadata["TCGA ID"] ==
                                 grade_metadata["TCGA ID"][idx]]["censored"]
            })
            metadata = pd.concat([metadata, df], axis=0)

    metadata_train, metadata_test = train_test_split(metadata, test_size=0.2)
    metadata_train.reset_index(inplace=True)
    metadata_test.reset_index(inplace=True)

    print(f"Saving Train CSV with {metadata_train.shape[0]} Samples")
    print(f"Saving Test CSV with {metadata_test.shape[0]} Samples")
    metadata_train.to_csv(configs['train_metadata'], index=False)
    metadata_test.to_csv(configs['test_metadata'], index=False)


def prepare_test_train(dataset_name, configs):

    if dataset_name == "tcga_brca":
        get_brca_dataset(configs['dataset_metadata'][dataset_name])
    elif dataset_name == "tcga_gbm_lgg_1024":
        get_gbm_lgg_dataset_1024(configs['dataset_metadata'][dataset_name])
    elif dataset_name == "tcga_gbm_lgg_512":
        get_gbm_lgg_dataset_512(configs['dataset_metadata'][dataset_name])
    elif dataset_name == "pathomic_kirc":
        pass
    else:
        raise NotImplementedError(f"The dataset {dataset_name} does not exist")


# dataset_name = "tcga_brca"
with open('configs.json') as json_file:
    configs = json.load(json_file)

dataset_name = "tcga_gbm_lgg_512"
prepare_test_train(dataset_name, configs)
dataset_name = "tcga_gbm_lgg_1024"
prepare_test_train(dataset_name, configs)
