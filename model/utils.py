import torch
from torchvision.ops import box_convert
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from scipy.optimize import linear_sum_assignment
from transformers import DetrImageProcessor

from dataset_utils.enums import Enums

def _init_device(model_device:str):
    
    if "cuda" in model_device:
        model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        model_device = torch.device("cpu")

    return model_device

def map_fine_labels_to_coarse(gt_fine_labels:torch.Tensor):
    """
    gt_fine_labels - shape is (total_labels) in a batch
    """
    
    def map_func(c):
        c = Enums.attr_id2label[c]
        c = Enums.NUSCENES_TO_GENERAL_CLASSES[c]
        return Enums.nuscenes_label2Id[c]
    
    coarse_labels = torch.tensor([map_func(x.item()) for x in gt_fine_labels], device=gt_fine_labels.device)
    return coarse_labels    

def query_aligned_loss(pred_coarse_logits:torch.Tensor, fine_labels:torch.Tensor, 
                vision_embeddings:torch.Tensor, fine_label_embeddings:torch.Tensor, 
                method:str="hungarian"):

    """
    pred_coarse_logits torch.Size([4, 100, 7])
    fine_labels torch.Size([10])
    vision_embeddings torch.Size([4, 100, 512])
    fine_label_embeddings torch.Size([10, 512])  
    
    So the loss is against assignments because they are query-aligned pseudo-labels, 
    while targets_fine are just object-level labels without query alignment.    
          
    """    

    coarse_labels = map_fine_labels_to_coarse(fine_labels) #(total_labels)
    bs = vision_embeddings.shape[0]
    
    prob = torch.nn.functional.softmax(pred_coarse_logits, -1)
    _, pred_coarse_labels = prob[..., :-1].max(-1)
    
    total_loss = 0.0
    total_assignments = []
    
    for b in range(bs):
        for gt_label_coarse in coarse_labels.unique():
            queries_mask = (pred_coarse_labels[b] == gt_label_coarse)
            queries = vision_embeddings[b][queries_mask]

            if queries.numel() == 0:
                continue

            # Collect fine labels for this coarse group
            fine_labels_mask = (coarse_labels == gt_label_coarse) #(total_labels)
            target_embeds = fine_label_embeddings[fine_labels_mask] #(total_labels, d)
            # target_fine = fine_labels[fine_labels_mask] # (K,)

            if target_embeds.shape[0] == 0:
                continue

            sim = torch.matmul(queries, target_embeds.T)
            M = queries.shape[0]

            # Greedy max similarity
            if method == "greedy":                
                assignments = sim.argmax(dim=-1)   # (M,)

            # Hungarian matching
            elif method == "hungarian":                
                sim_np = sim.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(-sim_np)
                
                assignments = torch.full((M,), -1, dtype=torch.long, device=sim.device)
                assignments[row_ind] = torch.tensor(col_ind, dtype=torch.long, device=sim.device)

            # ignore unassigned
            valid_mask = assignments != -1

            if valid_mask.any():

                loss = torch.nn.functional.cross_entropy(
                    sim[valid_mask],           # (M_valid, K)
                    assignments[valid_mask]    # (M_valid,)
                )
            else:
                loss = torch.tensor(0.0, device=queries.device)

            total_loss += loss

            total_assignments.append((b, queries_mask, assignments))

    return total_loss

def post_process_detr_outputs(image_processor:DetrImageProcessor, detr_outputs, target_sizes, threshold:float=0.25):

    processed_outputs = image_processor.post_process_object_detection(
        detr_outputs, target_sizes=target_sizes, threshold=threshold
    )

    return processed_outputs

def convert_detr_targets(targets):

    #TODO, replace the logic with 
    converted_targets = []
    for t in targets:
        size = t["orig_size"].tolist()  # (H, W)
        h, w = size

        boxes_abs = t["boxes"] * torch.tensor([w, h, w, h], device=t["boxes"].device)
        boxes_xyxy = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")

        converted_targets.append({
            "boxes": boxes_xyxy.cpu(),
            "labels": t["class_labels"].cpu()
        })

    return converted_targets

def compute_detr_metrics(metric:MeanAveragePrecision, predictions, targets):

    targets = convert_detr_targets(targets)
    for idx, preds in enumerate(predictions):
        predictions[idx] = {
            "boxes": preds["boxes"].cpu(),
            "scores": preds["scores"].cpu(),
            "labels": preds["labels"].cpu()
        }

    metric.update(predictions, targets)
