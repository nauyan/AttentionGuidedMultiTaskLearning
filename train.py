import json
from src.engine import Trainer

# TODO: Add to Configs
epochs = 1
# epochs = 100
# TODO: Add to Configs
tensorboard_logging = True


for dataset_name in ["gbmlgg", "kirc"]:
    if dataset_name == "gbmlgg":
        with open('configs_multitask_gbmlgg.json') as json_file:
            configs = json.load(json_file)
    elif dataset_name == "kirc":
        with open('configs_multitask_kirc.json') as json_file:
            configs = json.load(json_file)

    for nc_net_pretrained in [True, False]:

        for vit_pretrained in [True]:

            for imagenet_pretrained in [True]:

                for optimizer in ["radam"]:

                    for lr in [0.0001]:
                        configs['dataset_metadata']['optimizer_name'] = optimizer
                        configs['dataset_metadata']['learning_rate'] = lr
                        configs['model_configs']['cnn_encoder_pretrained'] = imagenet_pretrained
                        configs['model_configs']['vit_arch_pretrained'] = vit_pretrained
                        configs['model_configs']['nc_net_encoder'] = nc_net_pretrained


                        trainer = Trainer(configs)
                        trainer.train()

