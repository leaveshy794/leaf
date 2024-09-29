from pathlib import Path

import numpy as np
import cv2
import torch
import csv
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression,set_logging, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.torch_utils import select_device
import json
import os
import logging
from shutil import copy, rmtree
logger = logging.getLogger(__name__)


def merge_cut_pred(cut_total):
    """
    :param cut_total:
    :return:
    """
    mura_predict_result = {}
    for key in cut_total.keys():
        if len(cut_total[key]) == 1:
            mura_predict_result.setdefault(key, cut_total[key])
            continue
        else:
            mura_predict_result[key] = [cut_total[key][0]]
            for i in range(1, len(cut_total[key])):
                result = cut_total[key][i]
                conf = float(result[0])
                top, left, bottom, right = result[2], result[1], result[4], result[3]
                j = 0
                while j < len(mura_predict_result[key]):
                    predict = mura_predict_result[key][j]
                    predict_conf = float(predict[0])
                    predict_top, predict_left, predict_bottom, predict_right = predict[2], predict[1], predict[4], \
                        predict[3]
                    xmin, ymin, xmax, ymax = max(left, predict_left), max(top, predict_top), min(right,
                                                                                                 predict_right), min(
                        bottom, predict_bottom)
                    width, height = xmax - xmin, ymax - ymin
                    if width > 0 and height > 0:
                        xmin2 = min(left, predict_left)
                        ymin2 = min(top, predict_top)
                        xmax2 = max(right, predict_right)
                        ymax2 = max(bottom, predict_bottom)
                        new_conf = max(predict_conf, conf)
                        mura_predict_result[key][j] = [new_conf, xmin2, ymin2, xmax2, ymax2]
                        break
                    else:
                        j += 1
                else:
                    mura_predict_result[key].append([conf, left, top, right, bottom])
    return mura_predict_result


class Model():
    def __init__(self, color_spec, device="", gray_spec=None, common_spec=None):
        self.color_spec = color_spec
        self.gray_spec = gray_spec
        self.common_spec = common_spec
        self.augment = False
        self.conf_thres = 0.1
        self.iou_thres = 0.1
        self.agnostic_nms = False
        self.classes = None
        imgsz=(640,640)
        self.imgsz = imgsz
        weights = self.color_spec['model']
        self.device_num = "cuda:" + str(device)
        self.device = select_device(self.device_num)
        # Initialize
        set_logging()
        model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.model = model
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt


        # self.imgsz = imgsz
        self.half = False

    def create_result_dict(self):
        """
        create response dic main body
        :return:
        """
        result = {}
        result.setdefault("status", 200)
        result.setdefault("message", "Success")
        result.setdefault("result", [])
        return result.copy()

    def dump_error_result(self, code, message):
        """
        change status and insert error message
        :param code:
        :param message:
        :return:
        """
        self.result["status"] = code
        if not isinstance(message, str):
            message = repr(message)
        self.result["message"] = message

    def exchange_box(self, box, raw_h, raw_w):
        """
        Convert the values of the surrounding boxes in order required
        :param box:
        :param raw_h:
        :param raw_w:
        :return:
        """
        h_r = raw_h / 3640
        w_r = raw_w / 6040
        new_box_list = []
        for each_box in box:
            x1, y1, x2, y2 = each_box
            x1_n = int(x1 * w_r)
            y1_n = int(y1 * h_r)
            x2_n = int(x2 * w_r)
            y2_n = int(y2 * h_r)
            new_box_list.append([x1_n, y1_n, x2_n, y1_n, x2_n, y2_n, x1_n, y2_n])
        return new_box_list

    def unknown_predict(self, cut_img_unknown_dic, profile):
        """
        get the unknown prediction result
        :param source:
        :return:
        """
        defect_unknown = merge_cut_pred(cut_img_unknown_dic)
        # 准备好未知不良
        if len(defect_unknown.keys()) > 0:
            unk_total_defect = []
            for s in defect_unknown["unknown"]:
                unk_total_defect.append(["unknown", str(s[0]), s[1:]])
            # 置信度降序排序
            unk_b = sorted(unk_total_defect, key=lambda m: float(m[1]), reverse=True)
            unk_array = np.array(unk_b, dtype=object)
            # # 取置信度最高的5个结果
            code_list = [a[0] for a in unk_array[:5]]
            conf_list = [float(b[1]) for b in unk_array[:5]]
            box_list = [m[2] for m in unk_array[:5]]
            return code_list, conf_list, box_list
        else:
            return False

    def get_result(self, source, profile):
        """
        Read the picture in the specified path and perform the inference to get the original prediction result
        :param source:
        :return:
        """
        names = self.model.names
        raw_img = cv2.imread(str(source))

        raw_h, raw_w, c = raw_img.shape
        cut_img_label_dic = {}
        cut_img_unknown_dic = {}
        resize_img = cv2.resize(raw_img, (640, 640))
        img_array = np.array(resize_img)
        img_h, img_w, c = resize_img.shape
        imgsz = check_img_size(self.imgsz, s=self.stride)
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        bs=1
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:

                pred = self.model(im, augment=self.augment, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=3)
        # # 将图片转换为模型输入格式
        # img0 = img_array.transpose((2, 0, 1))[::-1]  # 从HWC转换到CHW并从BGR转换到RGB
        # img0 = np.ascontiguousarray(img0)
        # im = torch.from_numpy(img0.copy()).to(self.device)
        # im = im.half() if self.half else im.float()
        # im /= 255  # 归一化
        # if len(im.shape) == 3:
        #     im = im[None]
        # model_output = self.model(im, augment=self.augment, visualize=False)
        # pred = non_max_suppression(model_output, self.conf_thres, self.iou_thres, self.classes,
        #                            self.agnostic_nms, max_det=3)
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = [int(x.int()) for x in xyxy]
                        print(x1,y1,x2,y2)
                        bounds_x1, bounds_y1, bounds_x2, bounds_y2 = 116, 43, 171, 88
                        if (x1 >= bounds_x1 and y1 >= bounds_y1 and
                                x2 <= bounds_x2 and y2 <= bounds_y2):
                            defect_name = str(names[int(cls.int())])
                            print(defect_name, conf)
                            defect_name = str(names[int(cls.int())])
                            print(defect_name)
                            conf = format(float(conf.float()), '.4f')
                            if defect_name not in cut_img_label_dic.keys():
                                cut_img_label_dic[defect_name] = []
                            cut_img_label_dic[defect_name].append([float(conf), x1, y1, x2, y2])
                        unk_conf = float(profile['unknown']['CONFIDENCE'])


        defect = merge_cut_pred(cut_img_label_dic)
        print(defect)

        # 如果有检测到不良
        if len(defect.keys()) > 0:
            print(defect.keys())
            # 根据置信度阈值过滤结果
            whole_mura_defect = []
            for code in defect.keys():
                th = profile[code]['CONFIDENCE']
                for s in defect[code]:
                    if s[0] >= th:
                        whole_mura_defect.append([code, str(s[0]), s[1:]])
            print('====')
            print(whole_mura_defect)
            # 如果过滤后还是有不良
            if len(whole_mura_defect) > 0:
                b = sorted(whole_mura_defect, key=lambda m: float(m[1]), reverse=True)
                array = np.array(b, dtype=object)
                code_list = [a[0] for a in array[:10]]
                conf_list = [float(b[1]) for b in array[:10]]
                box_list = [m[2] for m in array[:10]]
                # box_list = self.exchange_box(box_list, raw_h, raw_w)
            # 如果过滤后没有不良
            else:
                code_list = ['WuJian']
                conf_list = ['1.0']
                box_list = [[0, 0, int(raw_w), 0, int(raw_w), int(raw_h), 0, int(raw_h)]]
                print(code_list)
        else:
            code_list = ['WuJian']
            print(code_list)
            conf_list = ['1.0']
            box_list = [[0, 0, int(raw_w), 0, int(raw_w), int(raw_h), 0, int(raw_h)]]
        # 如果目标检测算法没有检测到不良，则启用未知缺陷拦截算法
        if code_list[0] == 'WuJian':
            unk_res = self.unknown_predict(cut_img_unknown_dic, profile)
            if unk_res:
                code_list, conf_list, box_list = unk_res

                box_list = self.exchange_box(box_list, raw_h, raw_w)
                return code_list, conf_list, box_list
            else:
                return code_list, conf_list, box_list
        else:
            print(code_list)
            return code_list, conf_list, box_list

    def read_profile_preset_json(self):
        """
        read profile_preset.json
        :param :
        :return:
        """
        with open(self.color_spec['profile_preset'], 'r') as f:
            content = json.load(f)
        nums_class = len(self.model.names)
        if nums_class <= len(content.keys()):
            # print(content)
            return content
        else:
            self.dump_error_result(900, 'profile_preset.json is not complete')
            return False

    def read_profile(self, profile_path, profile_preset):
        """
        read profile csv and get each code priority and confidence threshold dic
        :param pred_result:
        :return:
        """
        profile = profile_preset
        # 如果配置文件不存在
        if not os.path.exists(profile_path):
            return profile
        else:
            try:
                with open(profile_path, encoding="UTF8") as f:
                    reader = csv.reader(f)
                    header_row = next(reader)
                    i0 = header_row.index('DEFECT_SIZE')
                    i1 = header_row.index('PRIORITY')
                    i2 = header_row.index('CONFIDENCE')
                    i3 = header_row.index('DEFECT_CODE')
                    for row in reader:
                        priority, threhold, defcet_name = row[i1], row[i2], row[i3]
                        if float(threhold):
                            profile[defcet_name]['CONFIDENCE'] = float(threhold)
                        if int(priority):
                            profile[defcet_name]['PRIORITY'] = int(priority)
                return profile
            except:
                return profile

    def get_final_result(self, pred_result, profile):
        """
        reasoning singel image final result with profile information
        :param pred_result:
        :return:
        """
        code_list, conf_list, box_list = pred_result
        final_result_dic = {}
        # if there is only one defect, image predict result is image final result
        if len(code_list) == 1:
            final_result_dic.setdefault("img_cls", code_list)
            final_result_dic.setdefault("img_box", box_list)
            final_result_dic.setdefault("img_score", conf_list)
        # if there is more than one defect,reasoning singel image final result with profile information
        else:
            res = [profile[code]['PRIORITY'] for code in code_list]
            loc = int(res.index(min(res)))
            final_result_dic.setdefault("img_cls", [code_list[loc]])
            final_result_dic.setdefault("img_box", [box_list[loc]])
            final_result_dic.setdefault("img_score", [conf_list[loc]])

        return final_result_dic

    def get_group_final(self, final_result_dic, gid, savepath, defect_count):
        """
        assemble task group result,in this task,group result is singel image final reuslt
        :param pred_result:
        :param gid:
        :param savepath:
        :return:
        """
        group_final_dic = {}
        group_final_dic.setdefault("img_cls", final_result_dic["img_cls"])
        group_final_dic.setdefault("img_box", final_result_dic["img_box"])
        group_final_dic.setdefault("img_score", final_result_dic["img_score"])
        group_final_dic.setdefault("gid", gid)
        group_final_dic.setdefault("defect", defect_count)
        group_final_dic.setdefault("type", "Final")
        group_final_dic.setdefault("savepath", savepath)
        return group_final_dic

    def check_files(self, img_json):
        """
        check image files bofore image inference
        :param img_json:
        :return:
        """

        #
        # check images
        def get_FileSize(filePath):
            filePath = str(filePath)
            fsize = os.path.getsize(filePath)
            fsize = fsize / float(1024 * 1024)
            return round(fsize, 2)

        try:
            img_info_list = img_json["image"]
            for each in img_info_list:
                img_path = str(each["path"])
                if not os.path.exists(img_path):
                    self.dump_error_result(611, 'Image not found')
                elif os.path.exists(img_path) and get_FileSize(img_path) > 0:
                    try:
                        raw_img = cv2.imread(img_path)
                        copy_img = raw_img.copy()
                    except:
                        self.dump_error_result(612, 'Image read error:%s' % (get_FileSize(img_path)))
                else:
                    continue
        except:
            self.dump_error_result(610, 'Image read error')

    def infer(self, img_json):
        """
        infrence main program
        :param img_json:
        :return: reulst(Format refer to the interface document)
        """
        # create inference response dic
        self.result = self.create_result_dict()
        try:
            logger.info('this is a test,task inference start')
            profile_preset = self.read_profile_preset_json()
            pattern_results_list = []
            img_info_list = img_json['image']
            self.check_files(img_json)
            if self.result['status'] == 200 and profile_preset != False:
                logger.info('running')
                profile = self.read_profile(img_json["info"]["profile_path"], profile_preset)
                print(profile)
                for each in img_info_list:
                    # in the task,only one image for each inference
                    img_path = each["path"]
                    # single image inference
                    pred_result = self.get_result(img_path, profile)
                    print("----")
                    print(pred_result)
                    final_result = self.get_final_result(pred_result, profile)
                    # print(final_result)
                    img_cls_value = final_result['img_cls'][0]
                    categories = {
                        'WuJian': 'WuJian',  # Adjust folder names as needed
                        'PS_lssue': 'PS_lssue',
                        'kaikou_Issue': 'kaikou_Issue',
                        'JiaoJie_Issue': 'JiaoJie_Issue'
                    }
                    unit_name = img_json['info']['UNIT_ID']
                    product_name = img_json['info']['GLASS_ID']
                    print(unit_name)

                    if img_cls_value in categories:
                        target_folder = categories[img_cls_value]
                        target_path = os.path.join('./save', unit_name, product_name, target_folder)

                        os.makedirs(target_path, exist_ok=True)
                        copy(img_path, os.path.join(target_path))
                    # assemble singel image result
                    pattern_results = {}
                    pattern_results.setdefault("img_cls", pred_result[0])
                    pattern_results.setdefault("img_box", pred_result[2])
                    pattern_results.setdefault("img_score", pred_result[1])
                    pattern_results.setdefault("uid", each['uid'])
                    pattern_results.setdefault("gid", each['gid'])
                    pattern_results.setdefault("defect", len(pred_result[0]))
                    pattern_results.setdefault("type", each["type"])
                    pattern_results.setdefault("savepath", img_json['info']['saveROOT_PATH'])
                    pattern_results.setdefault("final", final_result)
                    pattern_results_list.append(pattern_results)
                group_result = self.get_group_final(final_result, each['gid'], img_json['info']['saveROOT_PATH'],
                                                    len(pred_result[0]))
                pattern_results_list.append(group_result)
                # assemble result
                self.result["result"] = pattern_results_list
            logger.info('this is a test,task inference end')
        except Exception as e:
            # 捕获发生异常的代码行数
            # print(e.__traceback__.tb_lineno)
            self.dump_error_result(900, str(e))
        return self.result