import argparse
import json
import os
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a VOC-style annotation tree to COCO JSON.')
    parser.add_argument(
        '--voc-root',
        required=True,
        help='VOC root directory that contains XML annotation files.')
    parser.add_argument(
        '--output',
        required=True,
        help='Output COCO annotation JSON path.')
    return parser.parse_args()


def collect_categories(voc_root):
    categories = set()
    for root_dir, _, files in os.walk(voc_root):
        for xml_file in files:
            if not xml_file.endswith('.xml'):
                continue
            xml_path = os.path.join(root_dir, xml_file)
            root = ET.parse(xml_path).getroot()
            for obj in root.findall('object'):
                categories.add(obj.find('name').text)
    return sorted(categories)


def convert_voc_to_coco(voc_root):
    categories_list = collect_categories(voc_root)
    category_mapping = {
        cls_name: i + 1
        for i, cls_name in enumerate(categories_list)
    }

    coco = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': cls_id, 'name': cls_name}
            for cls_name, cls_id in category_mapping.items()
        ],
    }

    annotation_id = 1
    image_id = 1

    for root_dir, _, files in os.walk(voc_root):
        for xml_file in files:
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(root_dir, xml_file)
            root = ET.parse(xml_path).getroot()

            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            coco['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': width,
                'height': height,
            })

            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                cls_id = category_mapping[cls_name]

                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                box_width = xmax - xmin
                box_height = ymax - ymin

                coco['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': cls_id,
                    'bbox': [xmin, ymin, box_width, box_height],
                    'area': box_width * box_height,
                    'iscrowd': 0,
                })
                annotation_id += 1

            image_id += 1

    return coco, categories_list


def main():
    args = parse_args()
    coco, categories_list = convert_voc_to_coco(args.voc_root)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=4)

    print(
        'VOC multi-folder to COCO conversion finished! '
        f'Categories: {categories_list}')


if __name__ == '__main__':
    main()
