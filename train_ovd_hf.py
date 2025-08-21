import pytorch_lightning as pl
from pytorch_lightning import Trainer

import torch
from torch.utils.data import DataLoader

from dataset_utils.enums import Enums
from dataset_utils.nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset, NuScenesDETRPreprocessing

from model.model import OVDDetr
from model.utils import _init_device

class OVDDetrTrainer(pl.LightningModule):
    
    def __init__(self, model_kwargs:dict, dataset_kwargs:dict, trainer_kwargs:dict):
        super().__init__()

        self.model_kwargs = model_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.trainer_kwargs = trainer_kwargs
                
        self._init_model(model_kwargs)
                
        self.vision_lr = trainer_kwargs['vision_trainer_kwargs']['lr']
        self.vision_lr_backbone = trainer_kwargs['vision_trainer_kwargs']['lr_backbone']
        self.vision_weight_decay = trainer_kwargs['vision_trainer_kwargs']['weight_decay']
        
        self.text_lr = trainer_kwargs['text_trainer_kwargs']['lr']
        self.text_lr_backbone = trainer_kwargs['text_trainer_kwargs']['lr_backbone']
        self.text_weight_decay = trainer_kwargs['text_trainer_kwargs']['weight_decay']
        
        self.train_vision = trainer_kwargs['train_vision']
        self.train_text = trainer_kwargs['train_text']
        
    def _init_model(self, model_kwargs:dict):
        
        self.vision_model_device = _init_device(model_kwargs['vision_model_device'])
        self.text_model_device = _init_device(model_kwargs['text_model_device'])
    
        self.model = OVDDetr(
            vision_model=model_kwargs["vision_model"],
            text_model=model_kwargs["text_model"],
            vision_model_device=self.vision_model_device,
            text_model_device=self.text_model_device
        )
        
    def _init_dataloader(self, dataset_kwargs):

        dataset = NuScenesObjectDetectDataset(
            table_blob_paths=dataset_kwargs["table_blob_paths"],
            root_dir=dataset_kwargs["root_dir"]
        )    

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=dataset_kwargs["batch_size"],
            shuffle=dataset_kwargs["shuffle"],
            collate_fn=NuScenesDETRPreprocessing()
        )

        return dataloader

    def train_dataloader(self):
        return self._init_dataloader(self.dataset_kwargs["train_dataset_kwargs"])

    def val_dataloader(self):
        return self._init_dataloader(self.dataset_kwargs["eval_dataset_kwargs"])
    
    def configure_optimizers(self):
        
        param_dicts = []
        
        if self.train_vision:
            param_dicts.extend([{
                "params":[p for n, p in self.model.vision_model.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr":self.vision_lr
            },
            {
                "params":[p for n, p in self.model.vision_model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr":self.vision_lr_backbone
            }])
            
        if self.train_text:
            param_dicts.extend([{
                "params":[p for n, p in self.model.text_model.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr":self.text_lr
            },
            {
                "params":[p for n, p in self.model.text_model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr":self.text_lr_backbone
            }])
            
        optimizer = torch.optim.AdamW(param_dicts, weight_decay=self.vision_weight_decay)
        return optimizer
    
    def forward(self, data_items):        

        total_loss, coarse_loss, alignment_loss = self.model(**data_items)        
        return {
            "total_loss":total_loss, "coarse_loss":coarse_loss, "alignment_loss":alignment_loss
        }
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)        

        for k,v in loss_dict.items():
            self.log(f"Batch_Idx: {batch_idx} train_loss: {k}", v.item())

        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)        

        for k,v in loss_dict.items():
            self.log(f"Batch_Idx: {batch_idx} val_loss: {k}", v.item())

        return loss_dict['total_loss']
    
if __name__ == "__main__":
    
    model_kwargs = {
        "vision_model" : "facebook/detr-resnet-50",
        "text_model" : "openai/clip-vit-base-patch32",
        "vision_model_device" : f"cuda",
        "text_model_device" : f"cuda"
    }
    
    dataset_kwargs = {
        "train_dataset_kwargs" : {
            "table_blob_paths":['data/trainval01_blobs_US/tables.json'],
            "root_dir":'data/',
            "batch_size":4,
            "shuffle":False
        },
        "eval_dataset_kwargs" : {
            "table_blob_paths":['data/trainval01_blobs_US/tables.json'],
            "root_dir":'data',
            "batch_size":4,
            "shuffle":False
        }            
    }
    
    trainer_kwargs = {
        "train_vision":True,
        "train_text":True,
        "vision_trainer_kwargs":{
            "lr":1e-4, 
            "lr_backbone":1e-5, 
            "weight_decay":1e-4
        },
        "text_trainer_kwargs":{
            "lr":1e-4, 
            "lr_backbone":1e-5, 
            "weight_decay":1e-4                        
        }, 
        "min_epochs":10,
        "max_epochs":100,
        "gradient_clip_val":0.1, 
        "accumulate_grad_batches":8, 
        "log_every_n_steps":5
    }
    
    ovd_model = OVDDetrTrainer(
        model_kwargs=model_kwargs,
        dataset_kwargs=dataset_kwargs,
        trainer_kwargs=trainer_kwargs
    )
    
    trainer = Trainer(
        devices=1, 
        accelerator="cuda",
        gradient_clip_val=trainer_kwargs["gradient_clip_val"],
        accumulate_grad_batches=trainer_kwargs["accumulate_grad_batches"],
        log_every_n_steps=trainer_kwargs["log_every_n_steps"],
        min_epochs=trainer_kwargs["min_epochs"],
        max_epochs=trainer_kwargs["max_epochs"]
    )    
    
    trainer.fit(
        ovd_model
    )
    
    """
    TODO, 
    
    1. Complete train_detr_hf.py
    2. Implement evaluation metrics for the validation_step.
    3. Download remaining data splits.
    4. Commit and push code to github.
    """
