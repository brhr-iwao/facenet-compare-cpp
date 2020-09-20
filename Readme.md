# Facenet-compare-cpp

This is a C++ implementation of  [compare.py](https://github.com/davidsandberg/facenet/blob/master/src/compare.py) in [David Sandberg's Facenet](https://github.com/davidsandberg/facenet) but compares only two faces.

The code is inspired by [Terry Chan's MTCNN Face Recgnition system](https://github.com/Chanstk/FaceRecognition_MTCNN_Facenet)
and [Mandar Joshi's Facenet C++](https://github.com/mndar/facenet_classifier).

[OAID/FaceDetection](https://github.com/OAID/FaceDetection) is used for the Multi-Task CNN face detection.
Face detenction accuracy is improved over the previous release.

A pre-compiled binary which can be run on Windows (7 or later) without Python is placed in [the release](https://github.com/brhr-iwao/facenet-compare-cpp/releases).

#### Performance on LFW
Pre-trained models in  [davidsandberg/facenet](https://github.com/davidsandberg/facenet#pre-trained-models) are used.    
The face detector becomes more accurate than the previous release but [align_dataset_mtcnn.cpp](https://github.com/brhr-iwao/facenet-compare-cpp/blob/master/align_dataset_mtcnn.cpp]) even fails to detect only five faces (i.e. Darrell_Porter_0002.jpg, Ernie_Stewart_0001.jpg, Himmler_Rebu_0001.jpg, Marilyn_Monroe_0002.jpg and Muhammad_Saeed_al-Sahhaf_0003.jpg) of 13,233 [LFW](http://vis-www.cs.umass.edu/lfw) faces.

| Model name       |  AUC      |   EER   |  Threshold at EER  |   Accuracy at EER  |
|------------------|-----------|---------|--------------------|--------------------|
| [20170512-110547](https://drive.google.com/open?id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk)| 0.98086 | 0.04305 | 1.21726 | 0.95644 |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)| 0.98038 | 0.04906 | 0.57466 | 0.95093 |
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz)| 0.97531 | 0.05974 |  0.56109 | 0.94025 |

#### Details of face alignment and validation
##### Face alignment
1. Prepare your face dataset, i.e. download  [LFW dataset](http://vis-www.cs.umass.edu/lfw) and expand the archive.     
You can also use your own face dataset.
If you use your own dataset, the data structure must follow the LFW style,
i.e.face images of same person must be in one folder named FirstName_FamilyName
and these images must be named FirstName_FamilyName_0001.jpg, FirstName_FamilyName_0002.jpg and so on.
Images in further nested folder can not be recognized.

2. Run align_dataset_mtcnn.exe and align the images.     
You must specify the dataset folder, output folder in which to be stored aligned images, image size (a regular square)
and margin (must be same for upper, lower, left and right) as (edge) length in pixel
and extention of output format as follows:    
e.g. align_dataset_mtcnn.exe -i datasetFolder -o outputFolder -s 160 -m 32 -e png

3. If more than one face would be found in a image, the alignment may fail.

##### Validation
1. Run rocdata_on_lfw.exe and generate the validation data file.    
You must specify the model directory, model name (.pb), aligned face directory and extension (with period) of aligned face image file as follows:   
e.g. rocdata_on_lfw.exe models 20180402-114759.pb mtcnn_160 .png    
The validation data file named compulsory as "rocdata.txt" will be stored in the current directory ( same as rocdata_on_lfw.exe is).   
This file is a text file contains two columns.    
The left column: L2 distance between a pair of faces.    
The right column: 0 for a pair of same individuals or 1 for a pair of different.

2. Run the validate_with_rocdata.py.      
You must specify the validation data file as the argument:   
e.g. python validate_with_rocdata.py rocdata.txt    
Python, numpy, matplotlib and scikit-learn must be installed in advance for the calculation.
If the calculation is completed, the result will be displayed as such as:
```
AUC:  0.98086584    
EER from false positive rate:  0.0430574
EER from false negative rate:  0.0440587
Thershold at EER:  1.21726
Accuracy at EER:  0.9564419
```
The ROC (Reciever Operating Characteristic) curve image (png) will be saved in the current directory.
