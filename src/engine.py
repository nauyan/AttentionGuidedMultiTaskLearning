import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from torch.utils.data import DataLoader

from src.dataloader import TCGA_DL
from src.model import get_model
from src.utils import get_loss_fn, get_optimizer_fn, get_metric_fn, calculate_loss, calculate_metrics
from src.augmentation import get_augmentation
from src.meter import AverageValueMeter
class Trainer:

    def __init__(self, configs):

        self.model_dict = configs['model_configs']#model_dict # model_configs
        log_format = f"{configs['dataset_metadata']['task']}_{configs['dataset_metadata']['checkpoints_name']}_{configs['dataset_metadata']['optimizer_name']}_{configs['dataset_metadata']['learning_rate']}_{configs['model_configs']['cnn_encoder']}_Finetune_{configs['model_configs']['cnn_encoder_pretrained']}_{configs['model_configs']['vit_arch']}_Finetune_{configs['model_configs']['vit_arch_pretrained']}_NcNetEncoder_{configs['model_configs']['nc_net_encoder']}"

        self.writer = None
        if configs['tensorboard_logging']:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            writer_path = f"tensorboard_logs/{log_format}"
            self.writer = SummaryWriter(writer_path)

        self.task = configs['dataset_metadata']['task']
        self.num_classes = configs['dataset_metadata']['num_classes']
    
        train_loader = TCGA_DL(configs['dataset_metadata']['train_metadata'], configs['dataset_metadata']['train_images'],self.num_classes, self.task, get_augmentation())
        test_loader = TCGA_DL(configs['dataset_metadata']['test_metadata'], configs['dataset_metadata']['test_images'], self.num_classes, self.task, get_augmentation())

        self.train_dl = DataLoader(
            train_loader,
            batch_size=configs['dataloader_configs']['train_dataloader']['batch_size'],
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=None,
            pin_memory=True,
            drop_last=True,
        )

        self.test_dl = DataLoader(
            test_loader,
            batch_size=configs['dataloader_configs']['test_dataloader']['batch_size'],
            shuffle=False,#False,
            num_workers=os.cpu_count(),
            collate_fn=None,
            pin_memory=True,
            drop_last=True,
        )

        
        self.device = torch.device('cuda:0')
        self.model = get_model(configs['model_configs']['model_name'], self.num_classes, self.task, self.device, self.model_dict)
        # self.model = get_model("conv_trans_features", self.num_classes, self.task, self.device, self.model_dict)
        self.checkpoints_name = log_format


        self.epochs = configs['epochs']

        self.losses = [get_loss_fn(loss) for loss in configs['losses'].keys()]
        self.metrics = [get_metric_fn(evaluator) for evaluator in configs['evaluators'].keys()] # AUC_ROC Score, Accuracy, C-Index, cox_log_rank, accuracy_cox
        self.optimizer = get_optimizer_fn(configs['dataset_metadata']['optimizer_name'],
                                        configs['dataset_metadata']['learning_rate'], 
                                        self.model.parameters())

    def train(self):

        min_loss = 999
        max_accuracy = 0
        last_checkpoint = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = 0
            
            train_logs = {}
            loss_meters = {loss.__name__: AverageValueMeter() for loss in self.losses}
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
            
            for train_batch in tqdm(self.train_dl, total=len(self.train_dl)):
                self.model.train()
                out = self.model(
                    train_batch['feature_image'].to(self.device))

                # TODO: Add AverageMeter Loss Function  
                del train_batch["feature_image"]
                
                loss = 0
                for loss_fn in self.losses:
                    loss_value = loss_fn(self.device, **train_batch, **out)
                    loss_meters[loss_fn.__name__].add(loss_value)
                    loss = loss + loss_value
                loss_logs = {k: v.mean for k, v in loss_meters.items()}
                train_logs.update(loss_logs)

                loss_value = loss.item()
                train_loss = train_loss + loss_value

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                for metric_fn in self.metrics:
                    metric_value = metric_fn(**train_batch, **out)
                    if metric_value is not None:
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    else:
                        metrics_meters[metric_fn.__name__].add(metrics_meters[metric_fn.__name__].mean)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                train_logs.update(metrics_logs)

            
            test_logs = self.test()

            self.print_metrics_console(epoch, train_logs, test_logs, self.optimizer.param_groups[0]["lr"])
            self.log_metrics_tensorboard(epoch, train_logs, test_logs, self.optimizer.param_groups[0]["lr"])
            
            test_loss = [val for key, val in test_logs.items() if "loss" in key][0]

            if min_loss > test_loss:
                print("Saving Weights")
                torch.save(self.model, f"checkpoints/{self.task}_{self.checkpoints_name}_min_loss.pth")
                last_checkpoint = epoch
                min_loss = test_loss

            if epoch - last_checkpoint >= 25:
                self.optimizer.param_groups[0][
                    "lr"] = self.optimizer.param_groups[0]["lr"] * 0.75
                print("Reducing Learning rate to",
                      self.optimizer.param_groups[0]["lr"])
                last_checkpoint = epoch


        self.writer.flush()

    def test(self):
        with torch.no_grad():
            test_loss = 0

            logs = {}
            loss_meters = {loss.__name__: AverageValueMeter() for loss in self.losses}
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

            for test_batch in tqdm(self.test_dl, total=len(self.test_dl)):
                self.model.eval()
                out = self.model(
                    test_batch['feature_image'].to(self.device))


                del test_batch["feature_image"]
                
                loss = 0
                for loss_fn in self.losses:
                    loss_value = loss_fn(self.device, **test_batch, **out)
                    loss_meters[loss_fn.__name__].add(loss_value)
                    loss = loss + loss_value
                loss_logs = {k: v.mean for k, v in loss_meters.items()}
                logs.update(loss_logs)

                loss_value = loss.item()
                test_loss = test_loss + loss_value

                for metric_fn in self.metrics:
                    metric_value = metric_fn(**test_batch, **out)
                    if metric_value is not None:
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    else:
                        metrics_meters[metric_fn.__name__].add(metrics_meters[metric_fn.__name__].mean)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

            return logs

    def print_metrics_console(self, epoch, train_logs, test_logs, lr):
        # TODO: Populate using for loop

        print("*"*80)
        print(f"Epoch {epoch}")
        print(f"Train Logs {train_logs}")
        print(f"Test Logs {test_logs}")
        print(f"Learning Rate {lr}")
        print("*"*80)

    def log_metrics_tensorboard(self, epoch, train_logs, test_logs, lr):

        for key,value in train_logs.items():
            self.writer.add_scalar(f"{key}/train",value,epoch)

        for key,value in test_logs.items():
            self.writer.add_scalar(f"{key}/test",value,epoch)

        self.writer.add_scalar("Learning Rate", lr, epoch)

