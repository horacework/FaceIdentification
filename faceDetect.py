# coding=utf-8

# 利用摄像头采集人脸训练数据，保存到faceTrain中

import cv2
import numpy as np

catch_pic_num = 100         # 采集数量
catch_pic_type = True       # 训练样本是否为正样本，True为正样本，False为负样本

if __name__ == '__main__':

    cv2.namedWindow("test")
    cv2.namedWindow("test1")  # 命名一个新的窗口，这个主要实现的是，人脸在移动过程中，对其的移动路径进行标记。
    cap = cv2.VideoCapture(0)  # 打开1号摄像头
    success, frame = cap.read()  # 读取一桢图像，这个图像用来获取它的大小

    color = (134, 126, 255)  # 设置人脸框的颜色
    imagePath = "C:/Users/Administrator/Desktop/3.jpg"  # 测试图片路径
    classifierPath = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml"

    classfier = cv2.CascadeClassifier(classifierPath)  # 定义分类器，这里是opencv2自带的分类器
    success, frame = cap.read()
    # frame = cv2.imread(imagePath)
    # if success is not True:
    #     print("摄像头获取失败")
    #     exit()
    size = frame.shape[:2]  # 获得当前桢彩色图像的大小
    image = np.zeros(size, np.float16)  # 定义一个与当前桢图像大小相同的的灰度图像矩阵
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

    num = 1

    if catch_pic_type:
        path_name = "C:/Users/Administrator/Desktop/BossSensor-master/faceTrain/target"
    else:
        path_name = "C:/Users/Administrator/Desktop/BossSensor-master/faceTrain/other"

    while success:
        success, frame = cap.read()

        # frame = cv2.imread(imagePath)
        size = frame.shape[:2]  # 获得当前桢彩色图像的大小
        image = np.zeros(size, np.float16)  # 定义一个与当前桢图像大小相同的的灰度图像矩阵
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
        cv2.equalizeHist(image, image)  # 灰度图像进行直方图等距化
        # 如下三行是设定最小图像的大小
        divisor = 32
        h, w = size
        minSize = (w // divisor, h // divisor)
        faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)  # 人脸检测
        if len(faceRects) > 0:  # 如果检测到人脸，则将人脸进行标记
            for faceRect in faceRects:  # 对每一个人脸画圆形标记
                x, y, w, h = faceRect
                # cv2.rectangle(frame, (x, y), (x+w, y+h), color)

                # cv2.circle(img, (x + w // 2, y + h // 2), 2, color, 2, 8, 0) # 人脸追踪画点
                faceImage = frame[y:y + h, x:x + w]

                img_name = '%s/%d.jpg' % (path_name, num)
                # image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, faceImage)
                num += 1
                if num > catch_pic_num:  # 如果超过指定最大保存数量退出循环
                    break

                cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, color, 2, 8, 0)
                cv2.imshow("test1", faceImage)
        cv2.imshow("test", frame)  # 显示图像
        # cv2.imshow("test1", img)  # 显示人脸图像
        if num > catch_pic_num:
            break
        key = cv2.waitKey(10)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            break
    cv2.destroyAllWindows()
