import torch
from torch import nn
from transformers import AutoModelForSemanticSegmentation, AutoBackbone

from models.image.upernet import UperNetForSemanticSegmentation, UperNetHead
from models.image.hiera import Hiera


def get_model(model_cfg, n_classes):
    model_architecture = model_cfg["model_architecture"].lower()
    if model_architecture == "segformer":
        return SegFormerImageModel(model_cfg, n_classes)
    elif model_architecture == "upernet":
        return UpernetImageModel(model_cfg, n_classes)
    elif model_architecture == "hierasam_upernet":
        return HieraSamUpernetImageModel(model_cfg, n_classes)
    else:
        raise NotImplementedError()
    
    

class SegFormerImageModel(nn.Module):
    def __init__(self, model_cfg, n_classes):
        super().__init__()
        self.checkpoint_model = model_cfg["checkpoint_model"]
        self.frozen_encoder = model_cfg["frozen_encoder"]
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        MODEL_INITIALIZER = AutoModelForSemanticSegmentation
        model_args = {
            "pretrained_model_name_or_path": self.checkpoint_model,
            "ignore_mismatched_sizes": True,
            "num_labels": n_classes,
        }
        self.model = MODEL_INITIALIZER.from_pretrained(**model_args)
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")

    def forward(self, inputs):
        out = self.model(inputs)
        out = out.logits
        if out.shape[-2:] != inputs.shape[-2:]:
            out = nn.functional.interpolate(out, size=inputs.shape[-2:], mode=self.upsampling, align_corners=False)
        return out
    
    def encoder(self, inputs):
        outs = self.model.segformer(inputs, output_hidden_states=True).hidden_states
        return outs
    
    def decoder(self, feats):
        out = self.model.decode_head(feats)
        #if out.shape[-2:] != torch.Size([d*4 for d in feats[0].shape[-2:]]):
        #    out = nn.functional.interpolate(out, size=torch.Size([d*4 for d in feats[0].shape[-2:]]), mode="bilinear", align_corners=False)
        return out
    
    
class UpernetImageModel(nn.Module):
    def __init__(self, model_cfg, n_classes):
        super().__init__()
        self.checkpoint_model = model_cfg["checkpoint_model"]
        self.frozen_encoder = model_cfg["frozen_encoder"]
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        MODEL_INITIALIZER = UperNetForSemanticSegmentation
        model_args = {
            "pretrained_model_name_or_path": self.checkpoint_model,
            "ignore_mismatched_sizes": True,
            "num_labels": n_classes,
        }
        self.model = MODEL_INITIALIZER.from_pretrained(**model_args)
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")

    def forward(self, inputs):
        out = self.model(inputs)
        out = out.logits
        if out.shape[-2:] != inputs.shape[-2:]:
            out = nn.functional.interpolate(out, size=inputs.shape[-2:], mode=self.upsampling, align_corners=False)
        return out
    
    def encoder(self, inputs):
        outputs = self.model.backbone.forward_with_filtered_kwargs(
            inputs, output_hidden_states=True, output_attentions=False
        )
        features = outputs.feature_maps
        return features
    
    def decoder(self, feats):
        out = self.model.decode_head(feats)
        #if out.shape[-2:] != torch.Size([d*4 for d in feats[0].shape[-2:]]):
        #    out = nn.functional.interpolate(out, size=torch.Size([d*4 for d in feats[0].shape[-2:]]), mode="bilinear", align_corners=False)
        return out
    
    def forward_upto_last_feature(self, inputs):
        return self.model.forward_upto_last_feature(inputs)
    
    def decoder_lastfeat(self, feats):
        return self.model.decoder_lastfeat(feats)
    
    def forward_classifier(self, last_feature):
        out = self.model.forward_classifier(last_feature)
        #if out.shape[-2:] != torch.Size([d*4 for d in last_feature.shape[-2:]]):
        #    out = nn.functional.interpolate(out, size=torch.Size([d*4 for d in last_feature.shape[-2:]]), mode="bilinear", align_corners=False)
        return out
    

class HieraSamUpernetImageModel(nn.Module):
    def __init__(self, model_cfg, n_classes):
        super().__init__()
        size = model_cfg["checkpoint_model"]
        self.upsampling = model_cfg.get("upsampling", "bilinear")
        embed_dims = {"tiny": 96, "small": 96, "base_plus": 112}
        num_heads = {"tiny": 1, "small": 1, "base_plus": 2}
        stages = {"tiny": [1, 2, 7, 2], "small": [1,2,11,2], "base_plus": [2,3,16,3]}
        global_att_blocks = {"tiny": [5, 7, 9], "small": [7,10,13], "base_plus": [12,16,20]}
        window_pos_embed_bkg_spatial_size = {"tiny": [7, 7], "small": [7, 7], "base_plus": [14, 14]}
        upernet_size = {"tiny": 128, "small": 256, "base_plus": 512}
        self.backbone = Hiera(
            embed_dim=embed_dims[size],
            num_heads=num_heads[size],
            stages=stages[size],
            global_att_blocks=global_att_blocks[size],
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size[size]
        )
        
        checkpoint = torch.load(f"base_checkpoints/sam2_hiera_{size}.pt", weights_only=True)
        state_dict = checkpoint["model"]
        state_dict = {k.replace("image_encoder.trunk.", ""): v for (k, v)in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=False)
        self.hidden_size = [embed_dims[size], embed_dims[size]*2, embed_dims[size]*4, embed_dims[size]*8]

        self.head = UperNetHead(n_classes=n_classes, 
                                in_channels=self.hidden_size, 
                                channels=upernet_size[size])
    
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

    def forward(self, inputs):
        features = self.backbone(inputs)
        out = self.head(features)
        if out.shape[-2:] != inputs.shape[-2:]:
            out = nn.functional.interpolate(out, size=inputs.shape[-2:], mode=self.upsampling, align_corners=False)
        return out
    
    def encoder(self, inputs):
        features = self.backbone(inputs)
        return features
    
    def decoder(self, feats):
        out = self.head(feats)
        return out
    
    def decoder_lastfeat(self, features):
        last_feature = self.head.forward_upto_last_feature(features)
        return last_feature
    
    def forward_classifier(self, last_feature):
        output = self.head.forward_classifier(last_feature)
        return output