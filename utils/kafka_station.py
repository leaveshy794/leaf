#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   visualizer.py
@Time    :   2023/03/22 13:28:25
@Author  :   htx 
"""
import logging
import os
import copy
import time
import cv2
import json
import numpy as np


from uuid import uuid4
from confluent_kafka import Producer


class KafkaStation:
    """KafkaStation is to store temp value and send values to other location."""

    def __init__(self, name, kafka_address, task_id, topic, work_dir, round_up=6):
        """Store infos and pass to kafka.
        Args:
            name(str): name of kafka instance.
            kafka_address(str): ip address of kafka confluence.
            task_id(str): unique task id for each training or testing process.
            topic(str): topic of this task.
            work_dir(str):path to save files.
            round_up(int): number of decimal places.

        """
        self._infos = []
        self._kafka_address = kafka_address
        self._task_id = task_id
        self._topic = topic
        self._work_dir = work_dir
        os.makedirs(self._work_dir, exist_ok=True)
        self.producer = Producer({"bootstrap.servers": kafka_address})
        assert (
            isinstance(round_up, int) and round_up > 1
        ), "Round up value should be int and greater than 1."
        self.round_up = round_up
        self.logger = logging.getLogger(name)
        self._modules = dict(
            LINE=self.add_line_points,
            SHEET=self.add_sheet_values,
            COUNTER=self.add_count_value,
            GAUGE=self.add_gauge_value,
            RENDER_IMAGE=self.add_image,
        )

    def __getitem__(self, name):
        """Get item of add function."""
        return self._modules[name]

    def _generate_base_format(self, name, cat):
        """Base format of kafka message.
        Args:
            name(str): name of this message.
            cat(str):category of message, supports "LINE","DETAILS","RENDER_IMAGE","COUNTER","GAUGE".
        Return:
            dict of one message.
        """
        return dict(id=str(uuid4()), name=name, taskId=str(self._task_id), type=cat)

    def add_data(self, data_dict):
        """Add data to transfer station.
        each_data format
        For line: (graph_name, xAxisValue, xAxisName, yAxisValue, yAxisName) is need
        For sheet: (sheet_name, idx_name, headers, rows, mode="REPLACE") is need
        For count: (name, value) is need
        For gauge:(name, value) is need
        For image: (name, img) is need
        """
        assert isinstance(data_dict, dict)
        fn = self._modules[data_dict.pop("category")]
        fn(**data_dict)

    def add_data_list(self, data_list):
        """Add list of infos."""
        assert isinstance(data_list, list)
        for data in data_list:
            self.add_data(data)

    def add_line_points(
        self,
        graph_name,
        xAxisValue,
        xAxisName,
        yAxisValue,
        yAxisName=None,
    ):
        """Add each point to dot line

        Args:
            name(str): graph name
            xAxisValue(str): x axis value
            xAxisName(str): x axis name
            yAxisValue(float, int): value to store
            yAxisName(str, None): if None, yAxisName will be replace by name
        """
        if yAxisName is None:
            yAxisName = graph_name

        if not isinstance(yAxisValue, (int, float)):
            msg = (
                f"When adding line points, yAxisValue should be int or float,"
                f"but got type {type(yAxisValue)}, this value will ignore"
            )
            self.logger.warning(msg)
            return

        output = self._generate_base_format(graph_name, "LINE")
        output.update(
            data=dict(
                xAxisName=str(xAxisName),
                xAxisValue=str(xAxisValue),
                yAxisName=str(yAxisName),
                yAxisValue=round(yAxisValue, self.round_up),
            )
        )
        self._infos.append(output)

    def add_sheet_values(self, sheet_name, idx_name, headers, rows, mode="REPLACE"):
        """Add sheet.

        Args:
        sheet_name(str): sheet name.
        idx_name(str): representing different in different time.
        head(list[str]): colnames of sheet.
        rows(list[List[str]]): row values.
        mode(str): REPLACE OR APPEND, REPLACE will replace all values in sheet, accoring to name and idx_name.
        """
        assert mode in ["REPLACE", "APPEND"]
        output = self._generate_base_format(sheet_name, "DETAIL")
        new_rows = []
        for each_row in rows:
            new_each_row = []
            for x in each_row:
                if isinstance(x, str):
                    new_each_row.append(x)
                elif isinstance(x, (int, float)):
                    new_each_row.append(str(round(x, self.round_up)))
            new_rows.append(new_each_row)
        # assert is_list_of(headers, str), "Hears should be list of strs"
        output.update(
            data=dict(name=idx_name, mode=mode, headers=headers, rows=new_rows)
        )
        self._infos.append(output)

    def add_count_value(self, name, value):
        """Add value that is used to count number."""
        assert isinstance(value, (int, float))
        output = self._generate_base_format(name, "COUNTER")
        output.update(data=dict(increase=value))
        self._infos.append(output)

    def add_gauge_value(self, name, value):
        assert isinstance(value, (int, float))
        output = self._generate_base_format(name, "GAUGE")
        output.update(data=dict(value=value))
        self._infos.append(output)

    def add_image(self, name, img):
        """Image will be first save to disk, then pass the file path."""
        output = self._generate_base_format(name, "RENDER_IMAGE")
        assert isinstance(img, (np.ndarray, np.array)) and img.dtype == np.uint8
        img_name = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time)) + ".jpg"
        cv2.imwrite(os.path.join(self._work_dir, img_name), img)
        output.update(data=dict(value=os.path.join(self._work_dir, img_name)))
        self._infos.append(output)

    def clear(self):
        """Clear all infos"""
        self._infos = []

    def register_custom_fn(self, category, fn):
        def function_wrap(*args, **kwargs):
            name = kwargs["name"]
            category = kwargs["category"]
            output = self._generate_base_format(name, category)
            output.update(data=fn(*args, **kwargs))
            self._infos.append(output)

        if category in self._modules:
            raise ValueError("Category name {category} already exists")

        self._modules[category] = function_wrap

    def execute(self):
        """Send messsages."""

        def delivery_report(err, msg):
            """Called once for each message produced to indicate delivery result.
            Triggered by poll() or flush()."""
            if err is not None:
                self.logger.error("Kafka Message delivery failed: {}".format(err))

        for info in self._infos:
            self.logger.info(info)
            self.producer.poll(0)
            self.producer.produce(
                self._topic, json.dumps(info).encode("utf-8"), callback=delivery_report
            )
        message_num = self.producer.flush(2)

        if message_num > 0:
            self.logger.error(
                "Kafka Message delivery failed, stop send message by hand"
            )
        else:
            self.logger.debug(
                "Kafka Message delivery to address {}, topic {}".format(
                    self._kafka_address, self._topic
                )
            )

        self.clear()
