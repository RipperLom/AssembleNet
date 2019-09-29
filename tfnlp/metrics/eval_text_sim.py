#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 aibot.me, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: eval_text_sim.py
Author: wangyan
Date: 2019/09/06 11:53:24
Brief: 文本相似性度评估  pr曲线
"""
import os
import sys
import time
import numpy



class Sample(object):
    """
        单个样本的标注结果
    """
    def __init__(self):
        self.query = ""
        self.right = ""
        self.label = ""
        self.score = 0.0
        self.info  = ""
    
    def load(self, items):
        self.query = items[0]
        self.right = items[1].lower().strip()
        self.label = items[2].lower().strip()
        self.score = float(items[3])
        self.info = items[4]


class LabelPR(object):
    """
        评估：计算准确 召回 acc 和阈值间的关系
        画图
    """
    def __init__(self):
        self.data = []
        
    def process(self, path):
        self.load_data(path)
        
        t_list = []
        p_list = []
        r_list = []
        acc_list = []
        for threshold in numpy.linspace(0.5, 1.0, 50):
            P, R, ACC  = self.get_score(threshold)
            t_list.append(threshold)
            p_list.append(P)
            r_list.append(R)
            acc_list.append(ACC)
        
        data = {}
        data["阈值"] = self.fmt_float(t_list)
        data["准确率"] = self.fmt_float(p_list)
        data["召回率"] = self.fmt_float(r_list)
        data["正确率"] = self.fmt_float(acc_list)
        return data

    # 评估结果dump到文件
    def dump_file(self, data, file_name):
        outfile = open(file_name, "w")
        def get_str(items):
            return "\t".join([str(w) for w in items]) + "\n"
        
        outfile.write(get_str(data["阈值"]))
        outfile.write(get_str(data["准确率"]))
        outfile.write(get_str(data["召回率"]))
        outfile.write(get_str(data["正确率"]))
        outfile.close()
        return True
    
    # 评估结果图显示
    def show_html(self, data, file_name):
        from pyecharts.charts import Line
        import pyecharts.options as opts

        line = Line()
        line.add_xaxis([str(w) for w in data["阈值"]])
        line.add_yaxis(series_name="准确率", y_axis=data["准确率"], label_opts=opts.LabelOpts(is_show=False))
        line.add_yaxis(series_name="召回率", y_axis=data["召回率"], label_opts=opts.LabelOpts(is_show=False))
        line.add_yaxis(series_name="正确率", y_axis=data["正确率"], label_opts=opts.LabelOpts(is_show=False))

        line.set_global_opts(
                title_opts=opts.TitleOpts(title="对话系统效果"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                ),
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            )        
        line.render(file_name)
        return True


    def load_data(self, path):
        # query right label score other
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.rstrip("\r\n")
                if line.startswith("##"):
                    continue
                items = line.split("\t", 4)
                if len(items) != 5:
                    continue
                node = Sample()
                node.load(items)
                data_list.append(node)
                line = f.readline()
        self.data = data_list
        return True

    
    def fmt_float(self, score_list = []):
        return [ float("%.4f"%(f)) for f in score_list]
    

    def get_score(self, threshold):
        """
            P R Acc
        """
        TN = 0.0
        FP = 0.0
        FN = 0.0
        TP = 0.0
        for node in self.data:
            label = node.label
            if node.score < threshold:
                label = ""
            if label != "":
                if node.right == label:
                    TP += 1
                else:
                    FP += 1
            else:
                if node.right == label:
                    TN += 1
                else:
                    FN += 1
        P = TP / (TP + FP + 0.001)
        R = TP / (TP + FN + 0.001)
        ACC = (TP + TN)/(TP + FP + TN + FN)
        return (P, R, ACC)


def usage():
    """
    usage
    """
    print(sys.argv[0], "options")
    print("options")
    print("\tinput_path: 输入的文件名")
    print("\toutput_dir: 输出的路径")
    print("\toutput_name: 输出的文件名")


if __name__ == '__main__':
    # 输入：query right label score other
    # pr = LabelPR()
    # data = pr.process()
    # pr.dump_file(data, "tmp/eval_result.txt")
    # pr.show_html(data, "tmp/eval_result.html")

    print(len(sys.argv))
    if 4 != len(sys.argv):
        usage()
        sys.exit(-1)

    input_path = (sys.argv[1])
    output_dir = (sys.argv[2])
    output_name = (sys.argv[3])

    pr = LabelPR()
    data = pr.process(input_path)

    txt_path = os.path.join(output_dir, output_name + '.txt')
    html_path = os.path.join(output_dir, output_name + '.html')
    pr.dump_file(data, txt_path)
    pr.show_html(data, html_path)


