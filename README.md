# Face Identification by ML

Training model base on Keras Or Opencv


## Requirements

* WebCamera
* OpenCV 3.0
* Python 2.7
* Windows 10
* Nvidia graphics card and CUDA 8.0 (If you want to calculate faster)

Put images into faceTrain/target and faceTrain/other.
Also you can use faceDetect.py to collect facial images you need. It will put the images in to the right place automatically.

You can not only train the facial image but also train any object image whatever you want theoretically.

## Progress
* 2017-05-07 The code can train the model successfully. Through the test, the model is available.

## Usage
* Collect positive images and negative images.

```
$ python faceDetect.py
```


Second, Training the model.

```
$ python face_train_use_keras.py
```

Third, Use the newly trained model.

```
$ python faceDetectTest.py
```

Keras backend : Theano

## Install
See my blog for more details.[horacework.com](http://horacwork.com)
The installation and configuration of Keras is too complicated. Fuck!

## Licence

[MIT](https://github.com/Hironsan/BossSensor/blob/master/LICENSE)
