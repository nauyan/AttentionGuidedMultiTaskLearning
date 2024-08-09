import timm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import segmentation_models_pytorch as smp

# from HIPT.HIPT_4K.hipt_4k import HIPT_4K
# from HIPT.HIPT_4K.hipt_model_utils import eval_transforms

# from HIPT.HIPT_4K.hipt_model_utils import get_vit256
# from src.vision_transformer import get_vit256


class HIPT(torch.nn.Module):
    """
    HIPT Model (ViT_4K-256) for encoding non-square images (with [256 x 256] patch tokens), with 
    [256 x 256] patch tokens encoded via ViT_256-16 using [16 x 16] patch tokens.
    """

    def __init__(self, output_dim, arch):
        super(HIPT, self).__init__()
        model256_path = 'HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
        model256_path = None
        output_dim = {"vit_tiny": 192, "vit_small": 384, "vit_base": 768}
        self.model256 = get_vit256(pretrained_weights=model256_path,
                                   arch=arch)  #.to(device)
        self.classifier = nn.Linear(output_dim[arch], output_dim)

    def forward(self, x):
        x = x[:, :, 16:-16, 16:-16]
        features = self.model256(x)
        logits = self.classifier(features)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_hat, Y_prob


class NC_Net(nn.Module):

    def __init__(self, encoder, encoder_weights, device):
        super().__init__()
        self.model_name = "unet"
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.classes = 3
        self.activaton = None
        self.device = device

        self.model = None
        if self.model_name == "unet":
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=self.classes - 1,
                activation=self.activaton,
                decoder_attention_type="scse",
            )
            self.model1 = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=self.classes - 2,
                activation=self.activaton,
                decoder_attention_type="scse",
            )
        else:
            print("Model Not Found !")

    def forward(self, x):
        features = self.model.encoder(x)

        #         decoder_output = self.model.decoder(*features)
        #         decoder_output1 = self.model1.decoder(*features)

        #         decoder_output = self.model.segmentation_head(decoder_output)
        #         decoder_output1 = self.model1.segmentation_head(decoder_output1)

        #         masks = torch.zeros(
        #             decoder_output1.size(0),
        #             self.classes,
        #             decoder_output1.size(2),
        #             decoder_output1.size(3),
        #         ).to(self.device)
        #         masks[:, :2, :, :] = decoder_output
        #         masks[:, 2, :, :] = decoder_output1.squeeze()

        #         return masks

        return features

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


class TCGA(nn.Module):

    def __init__(self,
                 num_classes,
                 cnn_encoder="tf_efficientnet_b0_ns",
                 cnn_encoder_pretrained=False,
                 vit_arch="vit_tiny",
                 vit_arch_pretrained=False,
                 nc_net_encoder=False,
                 task="survival"):
        super().__init__()

        self.task = task
        # self.encoder = timm.create_model(
        #     "tf_efficientnet_b0_ns", pretrained=True, features_only=True
        # )
        # encoder_features = self.encoder.feature_info.channels()
        # tf_efficientnet_b5_ns

        #####################
        # self.encoder = NC_Net("tu-tf_efficientnet_b0_ns", None, torch.device('cpu'))
        if not nc_net_encoder:
            self.encoder = timm.create_model(cnn_encoder,
                                             pretrained=cnn_encoder_pretrained,
                                             features_only=True)
            encoder_features = self.encoder.feature_info.channels()
        else:
            print("Using NC-Net Encoder")
            self.encoder = timm.create_model(cnn_encoder,
                                             pretrained=cnn_encoder_pretrained,
                                             features_only=True)
            encoder_features = self.encoder.feature_info.channels()
            self.encoder = NC_Net(f"tu-{cnn_encoder}", None,
                                  torch.device('cpu'))
            self.encoder.load_state_dict(
                torch.load("NC-Net_tu-tf_efficientnet_b0_ns_all.pth"))
        #####################

        self.conv_block1 = nn.Sequential(nn.Conv2d(encoder_features[-4], 3, 1),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(encoder_features[-3], 3, 1),
                                         nn.ReLU())
        self.conv_block3 = nn.Sequential(nn.Conv2d(encoder_features[-2], 3, 1),
                                         nn.ReLU())
        self.conv_block4 = nn.Sequential(nn.Conv2d(encoder_features[-1], 3, 1),
                                         nn.ReLU())

        if vit_arch_pretrained:
            model256_path = 'HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
        else:
            model256_path = None
        self.model256 = get_vit256(pretrained_weights=model256_path,
                                   arch=vit_arch)

        output_dim = {"vit_tiny": 192, "vit_small": 384, "vit_base": 768}

        if self.task == "classification":
            # Add Condition for muli-scale to multiply by 4
            self.classifier = nn.Linear(output_dim[vit_arch] * 4, num_classes)
        elif self.task == "survival":
            self.hazard = nn.Sequential(
                nn.Linear(output_dim[vit_arch] * 4, 1), nn.Sigmoid())
            self.output_range = Parameter(torch.FloatTensor([6]),
                                          requires_grad=False)
            self.output_shift = Parameter(torch.FloatTensor([-3]),
                                          requires_grad=False)
        elif self.task == "multitask":
            self.classifier = nn.Linear(output_dim[vit_arch] * 4, num_classes)
            self.hazard = nn.Sequential(
                nn.Linear(output_dim[vit_arch] * 4, 1), nn.Sigmoid())
            self.output_range = Parameter(torch.FloatTensor([6]),
                                          requires_grad=False)
            self.output_shift = Parameter(torch.FloatTensor([-3]),
                                          requires_grad=False)

    def forward(self, x, features_only=False):
        x = self.encoder(x)  #[-1]

        c1 = self.conv_block1(x[-4])
        c2 = self.conv_block2(x[-3])
        c3 = self.conv_block3(x[-2])
        c4 = self.conv_block4(x[-1])

        C = [c1, c2, c3, c4]
        features = []
        for c in C:
            x = self.model256(c)
            features.append(x)

        if features_only:
            return features

        # TODO: Add Self Attention or Gated Attention to choose featuers
        x = torch.cat(features, axis=1)
        # print("Transformer Featuers",x.size())

        

        if self.task == "classification":
            logits = self.classifier(x)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            Y_prob = F.softmax(logits, dim=1)

            return {
                "hazard": None,
                "logits": logits,
                "Y_hat": Y_hat,
                "Y_prob": Y_prob
            }
        elif self.task == "survival":
            # logits = logits * self.output_range + self.output_shift
            hazard = self.hazard(x)

            return {
                "hazard": hazard,
                "logits": None,
                "Y_hat": None,
                "Y_prob": None
            }
        elif self.task == "multitask":
            hazard = self.hazard(x)
            logits = self.classifier(x)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            Y_prob = F.softmax(logits, dim=1)

            return {
                "hazard": hazard,
                "logits": logits,
                "Y_hat": Y_hat,
                "Y_prob": Y_prob
            }

    def get_last_selfattention(self, x):
        return self.model256.get_last_selfattention(x)
    
class TCGA_MULTI_TASK(nn.Module):

    def __init__():
        pass


from src import vision_transformer as vits


def get_vit256(pretrained_weights,
               arch='vit_small',
               device=torch.device('cuda:0')):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'
    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = True  # Initially set to False
    model256.eval()  # Uncomment
    model256.to(device)

    if pretrained_weights is None:
        return model256

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
        }
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(
            pretrained_weights, msg))

    return model256


def get_model(model_name, output_dim, task, device, model_dict):
    # TODO: Add Parameters for function if required
    if model_name == "hipt":
        model = HIPT(output_dim, "vit_base", task).to(device)
    elif model_name == "conv_trans":
        # model = TCGA(output_dim, "vit_small", task).to(device)
        model = TCGA(output_dim, model_dict['cnn_encoder'],
                     model_dict['cnn_encoder_pretrained'],
                     model_dict['vit_arch'], model_dict['vit_arch_pretrained'],
                     model_dict['nc_net_encoder'], task).to(device)
    else:
        raise NotImplementedError
    return model
