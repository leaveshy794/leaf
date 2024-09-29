import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm
from PIL import Image

classes = ["JiaoJie_Issue", "PS_lssue", "kaikou_Issue","Defect"]  # 类别

xml_path = "Defectxml"
txt_path = "labels1"
image_path = "Defect"

# 确保输出目录存在
if not os.path.exists(txt_path):
    os.makedirs(txt_path)


# 将原有的xmax, xmin, ymax, ymin换为x, y, w, h
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 输入时图像和图像的宽高
def convert_annotation(image_id, width, height):
    xml_file_path = os.path.join(xml_path, f'{image_id}.xml')
    txt_file_path = os.path.join(txt_path, f'{image_id}.txt')

    if not os.path.exists(xml_file_path):
        print(f'XML file not found for {image_id}, skipping.')
        return

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find('size')
        w = width
        h = height

        with open(txt_file_path, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    print(f'Class {cls} not in classes, skipping.')
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    except Exception as e:
        print(f'Error processing {image_id}: {e}')


if __name__ == "__main__":
    img_list = os.listdir(image_path)
    for img in tqdm(img_list):
        label_name = img[:-4]
        print(f'Processing {label_name}')
        image_full_path = os.path.join(image_path, img)

        if not os.path.exists(image_full_path):
            print(f'Image file not found for {label_name}, skipping.')
            continue

        try:
            with Image.open(image_full_path) as image:
                w, h = image.size
                print(f'Image {label_name} size: {w}x{h}')
                convert_annotation(label_name, w, h)
        except Exception as e:
            print(f'Error processing image {label_name}: {e}')