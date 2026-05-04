# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.runner import EpochBasedTrainLoop
from mmdet.registry import LOOPS
from mmengine.runner.amp import autocast

import torch
import numpy as np
import operator
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def update_cache(cache, pred, feature, score, shot_capacity):
    pred = int(pred)
    score = float(score)

    with torch.no_grad():
        item = [feature] + [score]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif item[1] > cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1), reverse=True)
        else:
            cache[pred] = [item]

def compute_cache_logits(feature, cache, alpha, beta, num_classes):
    device = feature.device

    with torch.no_grad():
        class_prototypes = []
        class_indices = sorted(cache.keys())
        for class_idx in class_indices:
            class_features = torch.stack([item[0] for item in cache[class_idx]])
            class_prototypes.append(class_features.mean(dim=0, keepdim=True))
        
        cache_keys = torch.cat(class_prototypes, dim=0).to(device)
        cache_values = F.one_hot(
                torch.tensor(class_indices, device=device),
                num_classes=num_classes
        ).float()
        
        affinity = feature @ cache_keys.t()  
        exp_affinity = torch.exp(beta * (affinity - 1))  
        cache_logits = exp_affinity @ cache_values  
        
        return alpha * cache_logits

@LOOPS.register_module()
class TTAODLoop(EpochBasedTrainLoop):
    """Loop for TTAOD."""
    def __init__(self, *args, shot_capacity=0, alpha=5.0, beta=5.0,
                thre_me=0.3, **kwargs):
        super().__init__(*args, **kwargs)

        # IDM 
        self.IDM_cache = {}
        self.shot_capacity = shot_capacity
        self.alpha, self.beta = alpha, beta
        self.thre_me = thre_me

        self.num_classes = len(self.dataloader.dataset.metainfo['classes'])


    def run(self):
        """Launch training."""
        self.runner.call_hook('before_train')

        for name, param in self.runner.model.student.named_parameters():
            param.requires_grad = False
            if 'prompt_embeddings' in name or 'tunable_linear' in name:
                param.requires_grad = True

        for name, param in self.runner.model.named_parameters():
            if param.requires_grad:
                print('Trainable param:', name)
        
        # IDM 
        if self.shot_capacity != 0:
            # self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
            # 改为本地加载，联网加载经常失败
            self.dinov2 = torch.hub.load("download/dinov2", "dinov2_vitl14", source='local', pretrained=False)
            self.dinov2.load_state_dict(torch.load('download/dinov2_vitl14_pretrain.pth'))
            self.dinov2.cuda().eval()

            self.dinov2_preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        assert self._max_epochs == 1, \
            'TTAODLoop assumes one-pass.'

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

        self.runner.call_hook('after_train')
        return self.runner.model

    
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')

        for idx, data_batch in enumerate(self.dataloader):
            # test phase
            if self.runner.val_loop is not None:
                data_batch_test = {
                    "inputs": data_batch["inputs"]["test"],
                    "data_samples": data_batch["data_samples"]["test"]
                }
                self.runner.model.student.eval()

                with autocast(enabled=self.runner.val_loop.fp16):
                    outputs = self.runner.model.val_step(data_batch_test)
                
                # IDM Enhancement
                if self.shot_capacity != 0:
                    for idx_, data_samples_ in enumerate(outputs):
                        if len(data_samples_.pred_instances) == 0:
                            continue

                        # 读取图片
                        img_path = data_samples_.img_path
                        img = Image.open(img_path).convert('RGB')
                        width, height = img.size

                        bboxes = data_samples_.pred_instances.bboxes.int()
                        scores = data_samples_.pred_instances.scores
                        labels = data_samples_.pred_instances.labels
                        cls_scores = data_samples_.pred_instances.cls_scores

                        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], 0, width)   # x1, x2
                        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], 0, height)  # y1, y2

                        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    
                        valid_mask = (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3]) & \
                                    (areas > 0) & (scores > self.thre_me)

                        valid_bboxes = bboxes[valid_mask]
                        valid_scores = scores[valid_mask]
                        valid_labels = labels[valid_mask]
                        valid_cls_scores = cls_scores[valid_mask]

                        if len(valid_bboxes) == 0:
                            continue
                        
                        # 批量裁剪图像
                        img_np = np.array(img)
                        cropped_imgs = []
        
                        for bbox in valid_bboxes.cpu().numpy():
                            x1, y1, x2, y2 = bbox
                            cropped_imgs.append(Image.fromarray(img_np[y1:y2, x1:x2]))
                            
                        processed_images = torch.stack([
                            self.dinov2_preprocess(crop_img) for crop_img in cropped_imgs
                        ]).cuda()
                        
                        with torch.no_grad():
                            image_features = self.dinov2(processed_images)   
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                        if self.IDM_cache:
                            cache_scores = []
                            for image_feature_ in image_features:
                                cache_logits = compute_cache_logits(image_feature_, self.IDM_cache, self.alpha, self.beta, self.num_classes)
                                cache_scores.append(cache_logits)

                            cache_scores = torch.stack(cache_scores, dim=0)
                            fusion_scores = valid_cls_scores + cache_scores
                            top_scores, top_labels = torch.topk(fusion_scores, k=1, dim=1)

                            labels[valid_mask] = top_labels.squeeze(1)
                            scores[valid_mask] = top_scores.squeeze(1)

                            outputs[idx_].pred_instances.labels = labels
                            outputs[idx_].pred_instances.scores = scores
                
                # 累积每个iter的评估结果
                self.runner.val_loop.evaluator.process(data_samples=outputs, data_batch=data_batch_test)
                self.runner.model.student.train()
            
            # teacher生成伪标签
            data_batch_teacher = {
                "inputs": data_batch["inputs"]["unsup_teacher"],
                "data_samples": data_batch["data_samples"]["unsup_teacher"]
            }
            data_batch_teacher = self.runner.model.data_preprocessor(data_batch_teacher, False)
            origin_pseudo_data_samples, _ = self.runner.model.get_pseudo_instances(
                data_batch_teacher['inputs'], data_batch_teacher['data_samples'])
            data_batch["data_samples"]['unsup_student'] = self.runner.model.project_pseudo_instances(
                    origin_pseudo_data_samples,
                    data_batch["data_samples"]['unsup_student'])
            
            # 维护IDM cache
            if self.shot_capacity != 0:
                for idx_, data_samples_ in enumerate(origin_pseudo_data_samples):
                    if len(data_samples_.gt_instances) == 0:
                        continue

                    # 读取图片
                    img_path = data_samples_.img_path
                    img = Image.open(img_path).convert('RGB')
                    width, height = img.size
                    
                    bboxes = data_samples_.gt_instances.bboxes.int()
                    scores = data_samples_.gt_instances.scores
                    labels = data_samples_.gt_instances.labels

                    bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], 0, width)   # x1, x2
                    bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], 0, height)  # y1, y2

                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    
                    valid_mask = (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3]) & \
                                (areas > 0)

                    valid_bboxes = bboxes[valid_mask]
                    valid_scores = scores[valid_mask]
                    valid_labels = labels[valid_mask]

                    if len(valid_bboxes) == 0:
                        continue

                    # 批量裁剪图像
                    img_np = np.array(img)
                    cropped_imgs = []
    
                    for bbox in valid_bboxes.cpu().numpy():
                        x1, y1, x2, y2 = bbox
                        cropped_imgs.append(Image.fromarray(img_np[y1:y2, x1:x2]))
                        
                    processed_images = torch.stack([
                        self.dinov2_preprocess(crop_img) for crop_img in cropped_imgs
                    ]).cuda()
                    
                    with torch.no_grad():
                        image_features = self.dinov2(processed_images)   
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # 更新IDM cache
                    for (score_, label_, image_feature_) in zip(valid_scores, valid_labels, image_features):
                        update_cache(self.IDM_cache, label_, image_feature_, score_, self.shot_capacity)
            
            # adaptation phase
            self.run_iter(idx, data_batch)
        
        # 计算测试集的最终评估结果
        metrics = self.runner.val_loop.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
