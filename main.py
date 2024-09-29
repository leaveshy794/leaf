import os
# import shutil

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import json

from torch.fx.experimental.unification.multipledispatch.dispatcher import source


from predictor import Model
# import pandas as pd
# from tqdm import tqdm
# import time
# import csv

def init(path):
    # 初始化推理器
    # 将权重文件加载到显存
    model_path = 'weights/best.pt'
    # model_path = 'runs/train/exp15/weights/best.pt'
    color_spec = {}
    color_spec['model'] = model_path
    color_spec['profile_preset'] = os.path.join(path, 'profile_preset.json')
    predictor = Model(color_spec, common_spec=None)
    return predictor


def predict(predictor, img_json):
    result_json = predictor.infer(img_json)
    return result_json



def read_txt(file):
    with open(file, 'r') as f:
        content = f.read().splitlines()
    return content


def read_json(file):
    with open(file, 'r') as f:
        content = json.load(f)
    return content

# 接口测试
if __name__ == "__main__":
    # 初始化推理器
    # 将权重文件加载到显存中
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    root_path = "D://Download//yolov5-7.0//yolov5-7.0"
    predictor = init(root_path)
    # 读取Json(假装收到请求)
    img_json_path = "D://Download//yolov5-7.0//yolov5-7.0//example.json"
    img_json = read_json(img_json_path)
    print(img_json)
    # 执行推理并组装响应结果
    result = predict(predictor,img_json)
    print('【推理结果】',result)