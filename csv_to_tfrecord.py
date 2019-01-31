# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 01:04:55 2018
@author: Xiang Guo
由CSV文件生成TFRecord文件
"""
 
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python csv_to_tfrecord.py --csv_input=data/train.csv  --output_path=data/train.record
  # Create test data:
  python csv_to_tfrecord.py --csv_input=data/test.csv  --output_path=data/test.record
"""
 
 
 
import os
import io
import pandas as pd
import tensorflow as tf
 
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
 
#os.chdir('C:\\Users\\Jupyter_notebook\\00_models\\research\\object_detection\\')
 
flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS
 
 
# TO-DO replace this with label map
#注意将对应的label改成自己的类别！！！！！！！！！！
def class_text_to_int(row_label):
    if row_label == 'laptop':
        return 1
    elif row_label == 'smallelectronicequipmen':
        return 2
    elif row_label == 'powerbank':
        return 3
    elif row_label == 'glassbottle':
        return 4
    elif row_label == 'winebottle':
        return 5
    elif row_label == 'umbrella':
        return 6
    elif row_label == 'metalcup':
        return 7
    elif row_label == 'lighter':
        return 8
    elif row_label == 'pressure':
        return 9
    elif row_label == 'drinkbottle':
        return 10
    elif row_label == 'scissor':
        return 11
    elif row_label == 'defibrillator':
        return 12
    elif row_label == 'gun':
        return 13
    elif row_label == 'magazine_clip':
        return 14
    elif row_label == 'fingerlock':
        return 15
    elif row_label == 'slingshot':
        return 16      
    elif row_label == 'expandablebaton':
        return 17
    elif row_label == 'zippooil':
        return 18
    elif row_label == 'nailpolish':
        return 19
    elif row_label == 'binghu':
        return 20
    elif row_label == 'handcuffs':
        return 21
    elif row_label == 'fireworks_crackers':
        return 22
    elif row_label == 'knife':
        return 23
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
 
 
def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
 
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
 
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
 
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
 
 
def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    #这里一定要改图片存储路径
    path = os.path.join(os.getcwd(), 'VOCdevkit\\VOC2007\\JPEGImages')
    examples = pd.read_csv(FLAGS.csv_input)#pd.read_csv(FLAGS.csv_input, encoding = 'GB2312')
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
 
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
 
 
if __name__ == '__main__':
    tf.app.run()
