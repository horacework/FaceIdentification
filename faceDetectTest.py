# coding=utf-8
import cv2
import numpy as np
from face_train_use_keras import Model

if __name__ == '__main__':

    modelPath = 'C:/Users/Administrator/Desktop/BossSensor-master/modelsss/man.face.model2.h5'

    # 加载模型
    model = Model()
    model.load_model(file_path=modelPath)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)
    classifierPath = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml"
    color = (134, 126, 255)  # 设置人脸框的颜色

    while True:
        _, frame = cap.read()  # 读取一帧视频

        # 图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(classifierPath)

        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        if len(faceRects) > 0:  # 如果检测到人脸，则将人脸进行标记
            for faceRect in faceRects:  # 对每一个人脸画圆形标记
                x, y, w, h = faceRect

                faceImage = frame[y:y + h, x:x + w]

                faceID = model.face_predict(faceImage)
                # 如果是“辣个男人”
                if faceID == 0:
                    # cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, color, 2, 8, 0)
                else:
                    pass

        cv2.imshow("test", frame)  # 显示图像

        key = cv2.waitKey(10)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            break

    cap.release()
    cv2.destroyAllWindows()
