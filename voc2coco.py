import os
import json
import xml.etree.ElementTree as ET

# VOC 根目录（包含多级子文件夹）
voc_root = "/root/autodl-tmp/VOC2007"  # 修改为你的 VOC 根目录
# 输出 COCO JSON 文件
coco_output = "coco_annotations.json"

# COCO 数据结构初始化
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# -----------------------
# 第一步：扫描所有 XML 文件提取类别
# -----------------------
categories_set = set()

for root_dir, dirs, files in os.walk(voc_root):
    for xml_file in files:
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(root_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            categories_set.add(cls_name)

# 给每个类别分配 COCO id
categories_list = sorted(list(categories_set))  # 排序方便管理
category_mapping = {cls_name: i+1 for i, cls_name in enumerate(categories_list)}

# 填充 COCO categories
for cls_name, cls_id in category_mapping.items():
    coco["categories"].append({
        "id": cls_id,
        "name": cls_name
    })

# -----------------------
# 第二步：生成 images 和 annotations
# -----------------------
annotation_id = 1  # COCO annotation ID 全局计数
image_id = 1       # COCO image ID 全局计数

for root_dir, dirs, files in os.walk(voc_root):
    for xml_file in files:
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(root_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 添加 image 信息
        coco["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # 处理每个 object
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            cls_id = category_mapping[cls_name]

            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            width_box = xmax - xmin
            height_box = ymax - ymin
            area = width_box * height_box

            # 添加 annotation 信息
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [xmin, ymin, width_box, height_box],
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

# -----------------------
# 第三步：保存 JSON
# -----------------------
with open(coco_output, 'w') as f:
    json.dump(coco, f, indent=4)

print(f"VOC multi-folder to COCO conversion finished! Categories: {categories_list}")