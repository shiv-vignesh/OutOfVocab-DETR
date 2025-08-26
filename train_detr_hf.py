import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import DetrForObjectDetection
from transformers import get_cosine_schedule_with_warmup

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset_utils.enums import Enums
from dataset_utils.nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset, NuScenesDETRPreprocessing, NuScenesDETRAugPreprocessing
from dataset_utils.weighted_sampler import compute_class_frequencies, compute_sample_weights

from model.utils import _init_device, post_process_detr_outputs, compute_detr_metrics

from logger import Logger

class DetrTrainer(pl.LightningModule):

    def __init__(self, model_kwargs:dict, dataset_kwargs:dict, trainer_kwargs:dict):
        super().__init__()
        
        self.model_kwargs = model_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.output_dir = trainer_kwargs["output_dir"]
        
        if not os.path.exists(trainer_kwargs["output_dir"]):
            os.makedirs(trainer_kwargs["output_dir"])
        
        self.logger_custom = Logger(trainer_kwargs["output_dir"])
        
        if model_kwargs["vision_model"] == "facebook/detr-resnet-50":        
            self.model = DetrForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=model_kwargs["vision_model"], 
                num_labels=len(Enums.nuscenes_Id2Labels),
                ignore_mismatched_sizes=True
            )

            self.vision_model_device = _init_device(model_kwargs['vision_model_device'])

        else:
            raise TypeError(f'Currently vision model can only be = facebook/detr-resnet-50 ')

        self.lr = trainer_kwargs['vision_trainer_kwargs']['lr']
        self.lr_backbone = trainer_kwargs['vision_trainer_kwargs']['lr_backbone']
        self.weight_decay = trainer_kwargs['vision_trainer_kwargs']['weight_decay']

        self.metric = MeanAveragePrecision(class_metrics=True)

    def _init_dataloader(self, dataset_kwargs, _type:str, _initialize_weighted_sampler:bool=True):
                    
        dataset = NuScenesObjectDetectDataset(
            table_blob_paths=dataset_kwargs["table_blob_paths"],
            root_dir=dataset_kwargs["root_dir"]
        )    
        
        if _type == "train" and _initialize_weighted_sampler:
            
            collate_fn = NuScenesDETRPreprocessing(
                apply_augmentation=dataset_kwargs.get("apply_augmentation", False)
            )
            
            class_counter, class_freq = compute_class_frequencies(dataset)
            sample_weights = compute_sample_weights(dataset, class_freq,
                                                    collate_fn.original_size,
                                                    use_class_weights_only=False)
            
            assert len(dataset) == len(sample_weights), f'Sample Weights {len(sample_weights)} and Dataset {len(dataset)} must be of equal lengths'
            
            sampler = WeightedRandomSampler(weights=sample_weights, 
                                            num_samples=len(sample_weights), 
                                            replacement=True)            
            
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=dataset_kwargs["batch_size"],
                collate_fn=collate_fn,
                sampler=sampler,
                shuffle=False
            )

        else:
            
            if dataset_kwargs["apply_mosaic_augmentation"]:
                collate_fn=NuScenesDETRAugPreprocessing(
                    apply_augmentation=dataset_kwargs.get("apply_augmentation", False)
                )                
                
                dataloader = DataLoader(
                                    dataset=dataset,
                                    batch_size=dataset_kwargs["batch_size"]*2, #each mosaic image has 4 tiles. so bs=32 is effectively 8. 
                                    shuffle=dataset_kwargs["shuffle"],
                                    collate_fn=collate_fn
                                )                
            
            else:
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=dataset_kwargs["batch_size"],
                    shuffle=dataset_kwargs["shuffle"],
                    collate_fn=NuScenesDETRPreprocessing(
                        apply_augmentation=dataset_kwargs.get("apply_augmentation", False)
                    )
                )

        return dataloader

    def train_dataloader(self):
        return self._init_dataloader(self.dataset_kwargs["train_dataset_kwargs"], _type="train", _initialize_weighted_sampler=False)

    def val_dataloader(self):
        return self._init_dataloader(self.dataset_kwargs["eval_dataset_kwargs"], _type="eval", _initialize_weighted_sampler=False)

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139        
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr":self.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]

        return torch.optim.AdamW(param_dicts, weight_decay=self.weight_decay)

        # optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        # num_training_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        # num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]        
    
    def forward(self, pixel_values:torch.Tensor, pixel_mask:torch.Tensor, labels:torch.Tensor, is_eval_mode:bool=False):

        labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        if is_eval_mode:
            return loss, loss_dict, outputs

        return loss, loss_dict

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.forward(batch["pixel_values"], 
                                       batch["pixel_mask"],
                                       batch["coarse_labels"])

        # logs metrics for each training_step, and the average across the epoch.
        if (batch_idx + 1) % 50 == 0:        
            self.logger_custom.log_message(f"Batch_Idx: {batch_idx} training_loss: {loss.item()}")
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            # Use your custom logger for readable logs
            self.logger_custom.log_message(f"[TRAIN] Epoch {self.current_epoch}, Step {self.global_step}, Batch {batch_idx}")
            for k, v in loss_dict.items():
                self.logger_custom.log_message(f"  {k}: {v.item():.4f}")

        return loss

    def validation_step(self, batch, batch_idx):

        loss, loss_dict, outputs = self.forward(batch["pixel_values"], 
                                       batch["pixel_mask"],
                                       batch["coarse_labels"], is_eval_mode=True)

        if (batch_idx + 1) % 100 == 0:
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.logger_custom.log_message(f"[VAL] Epoch {self.current_epoch}, Batch {batch_idx}")
            for k, v in loss_dict.items():
                self.logger_custom.log_message(f"  {k}: {v.item():.4f}")

        # target_sizes = [(pixel_values.shape[-2], pixel_values.shape[-1]) for pixel_values in batch["pixel_values"]]
        target_sizes = torch.tensor([img.shape[-2:] for img in batch["pixel_values"]], device=self.device)

        processed_outputs = post_process_detr_outputs(self.trainer.val_dataloaders.collate_fn.image_preprocessor, 
                                  outputs, target_sizes)

        compute_detr_metrics(self.metric, processed_outputs, batch['coarse_labels'])

        return loss

    def on_validation_epoch_end(self):

        results = self.metric.compute() 
        self.metric.reset()

        # log global metrics
        self.log("val/mAP", results["map"], prog_bar=True)
        self.log("val/mAP50", results["map_50"])
        self.log("val/mAP75", results["map_75"])
        self.log("val/mAR100", results["mar_100"])
        
        self.logger_custom.log_block(f"Validation metrics at epoch {self.current_epoch}")
        self.logger_custom.log_message(f"  mAP: {results['map']:.4f}")
        self.logger_custom.log_message(f"  mAP50: {results['map_50']:.4f}")
        self.logger_custom.log_message(f"  mAP75: {results['map_75']:.4f}")
        self.logger_custom.log_message(f"  mAR100: {results['mar_100']:.4f}")        

        # per-class metrics
        if results["map_per_class"] is not None:
            maps = results["map_per_class"]
            recalls = results["mar_100_per_class"]

            for i, m in enumerate(maps):
                class_label = Enums.nuscenes_Id2Labels[i]
                self.log(f"val/mAP/{class_label}", m)
                self.log(f"val/Recall/{class_label}", recalls[i])
                
                m_val = m.item() if not torch.isnan(m) else 0.0
                r_val = results["mar_100_per_class"][i].item() if not torch.isnan(results["mar_100_per_class"][i]) else 0.0                
                self.logger_custom.log_message(f"    {class_label}: mAP={m_val:.4f}, Recall={r_val:.4f}")                

if __name__ == "__main__":
    
    model_kwargs = {
        "vision_model" : "facebook/detr-resnet-50",
        "vision_model_device" : f"cuda",
    }

    dataset_kwargs = {
        "train_dataset_kwargs" : {
            "table_blob_paths":['data/trainval01_blobs_US/tables.json', 'data/trainval03_blobs_US/tables.json', 
                                'data/trainval04_blobs_US/tables.json', 'data/trainval05_blobs_US/tables.json'],
            "root_dir":'data/',
            "batch_size":32,
            "shuffle":True,
            "apply_augmentation":False,
            "apply_mosaic_augmentation":True
        },
        "eval_dataset_kwargs" : {
            "table_blob_paths":['data/trainval06_blobs_US/tables.json' ,'data/trainval07_blobs_US/tables.json'],
            "root_dir":'data',
            "batch_size":32,
            "shuffle":False,
            "apply_augmentation":False,
            "apply_mosaic_augmentation":False
        }            
    }

    trainer_kwargs = {
        "train_vision":True,
        "train_text":True,
        "vision_trainer_kwargs":{
            "lr":1e-4, 
            "lr_backbone":3e-5, 
            "weight_decay":1e-4
        },
        "min_epochs":10,
        "max_epochs":100,
        "gradient_clip_val":1.0,
        "accumulate_grad_batches":1, 
        "log_every_n_steps":5,
        "output_dir":"logs/detr_run_05_mosaic_augmentation"
    }        
    
    checkpoint_path="logs/detr_run_05_weighted_sampler/detr-epoch=20-val/mAP=0.2283.ckpt"
    
    if os.path.exists(checkpoint_path):
        detr_model = DetrTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_kwargs=model_kwargs,
            dataset_kwargs=dataset_kwargs,
            trainer_kwargs=trainer_kwargs            
        )

    else:
        detr_model = DetrTrainer(
            model_kwargs=model_kwargs,
            dataset_kwargs=dataset_kwargs,
            trainer_kwargs=trainer_kwargs
        )    

    checkpoint_cb = ModelCheckpoint(
        monitor="val/mAP",         # metric to monitor
        mode="max",                # maximize mAP
        save_top_k=1,              # save best model only
        filename="detr-{epoch:02d}-{val/mAP:.4f}", 
        save_last=True,
        dirpath=detr_model.output_dir
    )

    trainer = Trainer(
        devices=1, 
        accelerator="cuda",
        gradient_clip_val=trainer_kwargs["gradient_clip_val"],
        accumulate_grad_batches=trainer_kwargs["accumulate_grad_batches"],
        log_every_n_steps=trainer_kwargs["log_every_n_steps"],
        min_epochs=trainer_kwargs["min_epochs"],
        max_epochs=trainer_kwargs["max_epochs"],
        callbacks=[checkpoint_cb]
    )

    trainer.fit(
        detr_model
    )
