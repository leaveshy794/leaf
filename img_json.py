import os
import json

# 指定图片所在的文件夹路径
image_folder_path = "D://Download//yolov5-7.0//yolov5-7.0//download"

# 获取文件夹内所有图片文件的路径
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 创建一个空的列表来存储图片信息
images_info = []

# 遍历图片文件，为每张图片创建一个信息对象
for i, image_path in enumerate(image_files):
    image_info = {
        "path": image_path,
        "type": "color",
        "uid": f"{str(i).zfill(6)}",  # 生成一个格式化的 uid
        "gid": "0",
        "ATTR": {}
    }
    images_info.append(image_info)

# 创建一个包含模型信息和图片信息的 JSON 对象
data = {
    "image": images_info,
    "model": {"type": "AI"},
    "info": {
        "PRODUCT_ID": "P647F1FEB100",
        "UNIT_ID": "AOI800",
        "GLASS_ID": "768N3Z0025C4B",
        "model_dir": "D://Download//yolov5-7.0//yolov5-7.0//weights",
        "saveROOT_PATH": "D://Download//yolov5-7.0//yolov5-7.0",
        "profile_path": "./profile_cs.csv"
    }
}

# 将 JSON 对象转换为字符串
json_data = json.dumps(data, indent=4)

# 打印 JSON 字符串
print(json_data)

# 如果需要，可以将 JSON 数据保存到文件
with open('images_info.json', 'w') as json_file:
    json_file.write(json_data)