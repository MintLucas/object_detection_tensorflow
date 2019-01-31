import os
import cv2
import numpy as np
# 视频数据
# http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi
# http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets

# 标注信息
# http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top


def video2im(src, train_path='images/images_from_video/from_video1/train', test_path='images/images_from_video/from_video1/test', factor=2):
    """
    Extracts all frames from a video and saves them as jpgs
    """

    #os.mkdir(train_path)
    #os.mkdir(test_path)

    frame = 0
    cap = cv2.VideoCapture(src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Total Frame Count:', length)

    while True:
        check, img = cap.read()
        if check:
            if frame < 3600:
                path = train_path
            else:
                path = test_path

            img = cv2.resize(img, (1920 // factor, 1080 // factor))
            cv2.imwrite(os.path.join(path, str(frame) + ".jpg"), img)

            frame += 1
            print('Processed: ', frame, end='\r')

        else:
            break

    cap.release()


if __name__ == '__main__':
    video2im('D:/images/video/TownCentreXVID.avi')