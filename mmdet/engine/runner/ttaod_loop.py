# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.runner import EpochBasedTrainLoop
from mmdet.registry import LOOPS
from mmengine.runner.amp import autocast

import torch
import numpy as np
import operator
import random
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from mmengine.structures import InstanceData

def update_cache(cache, pred, feature, score, shot_capacity, image=None):
    pred = int(pred)
    score = float(score)

    with torch.no_grad():
        item = dict(
            feature=feature.detach(),
            score=score,
            image=image.copy() if image is not None else None)
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif item['score'] > cache[pred][-1]['score']:
                cache[pred][-1] = item
            cache[pred] = sorted(
                cache[pred], key=operator.itemgetter('score'), reverse=True)
        else:
            cache[pred] = [item]

def compute_cache_logits(feature, cache, alpha, beta, num_classes):
    device = feature.device

    with torch.no_grad():
        class_prototypes = []
        class_indices = sorted(cache.keys())
        for class_idx in class_indices:
            class_features = torch.stack(
                [item['feature'] for item in cache[class_idx]])
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


def bbox_iou(bbox, bboxes):
    if len(bboxes) == 0:
        return 0.0

    bboxes = np.asarray(bboxes, dtype=np.float32)
    x1 = np.maximum(bbox[0], bboxes[:, 0])
    y1 = np.maximum(bbox[1], bboxes[:, 1])
    x2 = np.minimum(bbox[2], bboxes[:, 2])
    y2 = np.minimum(bbox[3], bboxes[:, 3])

    inter_w = np.maximum(x2 - x1, 0)
    inter_h = np.maximum(y2 - y1, 0)
    inter = inter_w * inter_h
    area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0)
    bboxes_area = np.maximum(bboxes[:, 2] - bboxes[:, 0], 0) * np.maximum(
        bboxes[:, 3] - bboxes[:, 1], 0)
    union = area + bboxes_area - inter
    valid = union > 0
    if not valid.any():
        return 0.0
    return float(np.max(inter[valid] / union[valid]))

@LOOPS.register_module()
class TTAODLoop(EpochBasedTrainLoop):
    """Loop for TTAOD."""
    def __init__(self, *args, shot_capacity=0, alpha=5.0, beta=5.0,
                thre_me=0.3, memory_hallucination=True,
                hallucination_max_instances=3, hallucination_beta=1.0,
                hallucination_iou_thr=0.2, hallucination_max_trials=10,
                hallucination_scale_range=(0.5, 1.5), **kwargs):
        super().__init__(*args, **kwargs)

        # IDM 
        self.IDM_cache = {}
        self.shot_capacity = shot_capacity
        self.alpha, self.beta = alpha, beta
        self.thre_me = thre_me
        self.memory_hallucination = memory_hallucination
        self.hallucination_max_instances = hallucination_max_instances
        self.hallucination_beta = hallucination_beta
        self.hallucination_iou_thr = hallucination_iou_thr
        self.hallucination_max_trials = hallucination_max_trials
        self.hallucination_scale_range = hallucination_scale_range

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

    def _raw_inputs_are_bgr(self):
        data_preprocessor = self.runner.model.data_preprocessor
        if hasattr(data_preprocessor, 'data_preprocessor'):
            data_preprocessor = data_preprocessor.data_preprocessor
        return getattr(data_preprocessor, 'bgr_to_rgb', False)

    def _sample_memory_instance(self):
        valid_classes = [
            class_idx for class_idx, items in self.IDM_cache.items()
            if any(item.get('image') is not None for item in items)
        ]
        if not valid_classes:
            return None, None

        class_idx = random.choice(valid_classes)
        candidates = [
            item for item in self.IDM_cache[class_idx]
            if item.get('image') is not None
        ]
        return class_idx, random.choice(candidates)

    def _paste_memory_instance(self, image_tensor, instance_image,
                               placed_bboxes):
        _, img_h, img_w = image_tensor.shape
        inst_w, inst_h = instance_image.size
        if inst_w <= 0 or inst_h <= 0 or img_w <= 1 or img_h <= 1:
            return None

        scale_min, scale_max = self.hallucination_scale_range
        scale = random.uniform(scale_min, scale_max)
        new_w = max(int(round(inst_w * scale)), 2)
        new_h = max(int(round(inst_h * scale)), 2)
        fit_scale = min((img_w - 1) / new_w, (img_h - 1) / new_h, 1.0)
        new_w = max(int(round(new_w * fit_scale)), 2)
        new_h = max(int(round(new_h * fit_scale)), 2)
        if new_w >= img_w or new_h >= img_h:
            return None

        bbox = None
        for _ in range(self.hallucination_max_trials):
            x1 = random.randint(0, img_w - new_w)
            y1 = random.randint(0, img_h - new_h)
            candidate = [x1, y1, x1 + new_w, y1 + new_h]
            if bbox_iou(candidate, placed_bboxes) <= self.hallucination_iou_thr:
                bbox = candidate
                break
        if bbox is None:
            return None

        resized = instance_image.resize((new_w, new_h), Image.BILINEAR)
        instance_np = np.asarray(resized, dtype=np.float32)
        if self._raw_inputs_are_bgr():
            instance_np = instance_np[..., ::-1].copy()
        instance_tensor = torch.from_numpy(instance_np).permute(2, 0, 1)
        instance_tensor = instance_tensor.to(
            device=image_tensor.device, dtype=torch.float32)

        lam = float(np.random.beta(self.hallucination_beta,
                                   self.hallucination_beta))
        x1, y1, x2, y2 = bbox
        image_region = image_tensor[:, y1:y2, x1:x2].float()
        mixed_region = (1 - lam) * image_region + lam * instance_tensor
        if not image_tensor.dtype.is_floating_point:
            mixed_region = mixed_region.round().clamp(0, 255)
        image_tensor[:, y1:y2, x1:x2] = mixed_region.to(image_tensor.dtype)
        return bbox

    def _apply_memory_hallucination(self, inputs, data_samples):
        if (self.shot_capacity == 0 or not self.memory_hallucination
                or not self.IDM_cache):
            return

        if isinstance(inputs, torch.Tensor):
            batch_inputs = [inputs[i] for i in range(inputs.shape[0])]
        else:
            batch_inputs = inputs

        for image_tensor, data_sample in zip(batch_inputs, data_samples):
            gt_instances = data_sample.gt_instances
            if len(gt_instances) != 0:
                if ('scores' not in gt_instances or
                        (gt_instances.scores >=
                         self.runner.model.semi_train_cfg.cls_pseudo_thr).any()):
                    continue

            max_instances = min(self.hallucination_max_instances,
                                sum(len(v) for v in self.IDM_cache.values()))
            if max_instances <= 0:
                continue

            num_instances = random.randint(1, max_instances)
            placed_bboxes, labels, scores = [], [], []
            for _ in range(num_instances):
                label, memory_item = self._sample_memory_instance()
                if memory_item is None:
                    break

                bbox = self._paste_memory_instance(
                    image_tensor, memory_item['image'], placed_bboxes)
                if bbox is None:
                    continue

                placed_bboxes.append(bbox)
                labels.append(label)
                scores.append(memory_item['score'])

            if not placed_bboxes:
                continue

            gt_instances = InstanceData()
            gt_instances.bboxes = torch.tensor(
                placed_bboxes, dtype=torch.float32, device=image_tensor.device)
            gt_instances.labels = torch.tensor(
                labels, dtype=torch.long, device=image_tensor.device)
            gt_instances.scores = torch.tensor(
                scores, dtype=torch.float32, device=image_tensor.device)
            data_sample.gt_instances = gt_instances

    
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

                with torch.no_grad():
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
                    for (score_, label_, image_feature_, cropped_img_) in zip(
                            valid_scores, valid_labels, image_features,
                            cropped_imgs):
                        update_cache(self.IDM_cache, label_, image_feature_,
                                     score_, self.shot_capacity, cropped_img_)
            
            self._apply_memory_hallucination(
                data_batch["inputs"]['unsup_student'],
                data_batch["data_samples"]['unsup_student'])
            
            # adaptation phase
            self.run_iter(idx, data_batch)
        
        # 计算测试集的最终评估结果
        metrics = self.runner.val_loop.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
