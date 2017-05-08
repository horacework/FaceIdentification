# Face Identification by ML

Training model base on Keras and Opencv


## Requirements

* WebCamera
* OpenCV 3.0
* Python 2.7
* Windows 10
* Nvidia graphics card and CUDA 8.0 (If you want to calculate faster)

Put images into [faceTrain/target](https://github.com/horacework/FaceIdentification/tree/master/faceTrain/target) and [faceTrain/other](https://github.com/horacework/FaceIdentification/tree/master/faceTrain/other).
Also you can use faceDetect.py to collect facial images you need. It will put the images in to the right place automatically.

You can not only train the facial image but also train any object image whatever you want theoretically.

In my repository, the two training folders contains 98 pictures each and [modelsss](https://github.com/horacework/FaceIdentification/tree/master/modelsss) has two model files I had trained.

## Progress
* 2017-05-07 The code can train the model successfully. Through the test, the model is available.

## Usage
* Collect positive images and negative images.

```
$ python faceDetect.py
```


* Training the model.

```
$ python face_train_use_keras.py
```

* Use the newly trained model.

```
$ python faceDetectTest.py
```

* Keras backend : Theano

## Install
See my blog for more details. [horacework.com](http://horacwork.com)

The installation and configuration of Keras is too complicated. 

Fuck!

## Reference
[Hironsan/BossSensor](https://github.com/Hironsan/BossSensor)

[人脸检测及识别python实现系列](http://www.cnblogs.com/neo-T/p/6477378.html)

## Licence
[MIT](https://github.com/horacework/FaceIdentification/blob/master/LICENSE)
