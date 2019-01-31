# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:52:02 2019
@author: Lucas
append random creade train and test dataset
path use .\
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import random

#chdir添加working dir
#os.chdir('C:\\Users\\Jupyter_notebook\\00_models\\research\\object_detection\\images\\test')
path = '.\\VOCdevkit\\VOC2007\\Annotations'
#path = '.\\training\\images\\train'
xml_list = os.listdir(path)
total_xml = [path + '\\' + i for i in xml_list if i.endswith('.xml')]
total_num = len(total_xml)

train_persent = 0.9
train_num = int(total_num*train_persent)

#列表里随机取train_num个值并打乱
train_xml = random.sample(total_xml, train_num)
train_addr = 'data\\train.csv'
test_addr = 'data\\test.csv'

train_xml_to_pd = []
test_xml_to_pd = []

def xml_to_csv(path):
    xml_list = []
    #对path里的所有xml文件
    #for xml_file in glob.glob(path + '/*.xml'):
    for xml_file in total_xml:
        print(xml_file)
        tree = ET.parse(xml_file)#用xml.etree打开xml
        root = tree.getroot()
        for member in root.findall('object'):
            value = [root.find('filename').text,#找到<filename>对应行
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,#找到<name>即我们标记的类别
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     ]
            #做处理前把tutple变为list
            #由于文件filename行命名不规范，需要去掉路径头文件只要后几行的文件名
            value[0] = value[0][-29:]
            if xml_file in train_xml:
                train_xml_to_pd.append(value)#都存入一个List里
            else:
                test_xml_to_pd.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_xml_df = pd.DataFrame(train_xml_to_pd, columns=column_name)#写成行列csv文件
    test_xml_df = pd.DataFrame(test_xml_to_p    d, columns=column_name)
    return train_xml_df, test_xml_df


def main():
    image_path = path
    train_xml_df, test_xml_df = xml_to_csv(image_path)#一个dataframe文件
    train_xml_df.to_csv(train_addr, index=None)
    test_xml_df.to_csv(test_addr, index=None)
    print('Successfully converted xml to csv.')


main()