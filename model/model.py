import torch
from transformers import CLIPTextModel, DetrForObjectDetection

from dataset_utils.enums import Enums
from model.utils import query_aligned_loss

class OVDDetr(torch.nn.Module):
    
    def __init__(self, 
                vision_model:str="facebook/detr-resnet-50",
                text_model:str="openai/clip-vit-base-patch32",
                vision_model_device:torch.device=torch.device("cpu"), 
                text_model_device:torch.device=torch.device("cpu")):
        
        super(OVDDetr, self).__init__()
        
        if vision_model == "facebook/detr-resnet-50":
            self.vision_model = DetrForObjectDetection.from_pretrained(vision_model,
                                                                       num_labels=len(Enums.nuscenes_Id2Labels),
                                                                       ignore_mismatched_sizes=True)
        
        else:
            raise TypeError(f'Currently vision model can only be = facebook/detr-resnet-50 ')
        
        if text_model == "openai/clip-vit-base-patch32":
            self.text_model = CLIPTextModel.from_pretrained(text_model)
            
        else:
            raise TypeError(f'Currently text model can only be = openai/clip-vit-base-patch32 ')            
        
        #TODO, add layernorm
        self.proj_layer = torch.nn.Linear(self.vision_model.model.config.d_model, self.text_model.config.hidden_size)
        
        self.vision_model_device = vision_model_device
        self.text_model_device = text_model_device
    
    def forward_vision_model(self, pixel_values:torch.Tensor, pixel_mask:torch.Tensor, 
                coarse_labels:torch.Tensor):

        outputs = self.vision_model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=coarse_labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        last_hidden_state = outputs.last_hidden_state
        
        pred_coarse_labels = outputs.logits

        return loss, loss_dict, pred_coarse_labels,last_hidden_state, outputs

    def forward_text_model(self, fine_labels_input_ids:torch.Tensor, fine_labels_attention_mask:torch.Tensor):                

        text_embeddings = self.text_model(fine_labels_input_ids, fine_labels_attention_mask).pooler_output
        return text_embeddings

    def forward(self, pixel_values:torch.Tensor, pixel_mask:torch.Tensor,
                fine_labels_input_ids:torch.Tensor, 
                fine_labels_attention_mask:torch.Tensor,
                coarse_labels:torch.Tensor=None, fine_labels:torch.Tensor=None, is_eval_mode:bool=False):

        text_embeddings = self.forward_text_model(fine_labels_input_ids.to(self.text_model_device), 
                                                fine_labels_attention_mask.to(self.text_model_device))
        
        #vision_embeddings = (bs, num_queries=100, 256)
        #pred_coarse_labels = (bs, num_queries=100, num_labels)
        coarse_labels = [{k: v.to(self.vision_model_device) for k, v in t.items()} for t in coarse_labels]
        coarse_loss, coarse_loss_dict, pred_coarse_labels, vision_embeddings, outputs = self.forward_vision_model(pixel_values.to(self.vision_model_device), 
                                                                                                        pixel_mask.to(self.vision_model_device), 
                                                                                                        coarse_labels)
        vision_embeddings = self.proj_layer(vision_embeddings)

        alignment_loss = query_aligned_loss(
            pred_coarse_labels, 
            fine_labels, 
            vision_embeddings, 
            text_embeddings
        )
        
        total_loss = coarse_loss + 0.5 * alignment_loss

        if is_eval_mode:
            return total_loss, coarse_loss, alignment_loss, outputs
        
        return total_loss, coarse_loss, alignment_loss
