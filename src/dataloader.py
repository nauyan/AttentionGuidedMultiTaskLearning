import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io
from glob import glob

class TCGA_DL(Dataset):

    def __init__(self,
                 metadata,
                 images_path,
                 num_classes,
                 task,
                 data_augmentation=None):

        self.metadata = pd.read_csv(metadata)
        self.num_classes = num_classes
        self.augmentation = data_augmentation
        # TODO: Add functionality to dataloader to select label depending on task
        # TODO: Add assertion for num_classes == num_classes assessed from Dataframe
        self.task = task

        self.images = np.load(images_path)
        

    def __getitem__(self, idx):
        file_path = self.metadata['file_path'][idx]

        feature_image = self.images[idx]

        if self.augmentation:
            sample = self.augmentation(image=feature_image)
            feature_image = sample["image"]

        feature_image = feature_image / 255.0
        feature_image = torch.from_numpy(feature_image).permute(2, 0,
                                                                1).float()

        grade = self.metadata['classification_label'][idx]
        grade = F.one_hot(torch.as_tensor(int(grade)), self.num_classes)

        censor = 1 - self.metadata['censored'][idx]

        surival_time = self.metadata['survival_time'][idx]

        return {
            "file_path": file_path,
            "feature_image": feature_image,
            "grade": grade,
            "censor": censor,
            "survtime": surival_time,
            "num_classes": self.num_classes
        }

    def __len__(self):
        # return 100
        return self.metadata.shape[0]

    def preprocessing(self):
        # TODO: Implement that converts data to torch tensors and move to gpu we need to move data to gpu
        pass

class TCGA_DL_New(Dataset):

    def __init__(self,
                 dataset_dir=None):

        self.paths = glob(f"{dataset_dir}/*.pt")
        

    def __getitem__(self, idx):
        return torch.load(self.paths[idx])

    def __len__(self):
        # return 100
        return len(self.paths)

    def preprocessing(self):
        # TODO: Implement that converts data to torch tensors and move to gpu we need to move data to gpu
        pass
