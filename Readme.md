# Facenet-compare-cpp

This is a C++ implementation of  [compare.py](https://github.com/davidsandberg/facenet/blob/master/src/compare.py)  in [David Sandberg's Facenet](https://github.com/davidsandberg/facenet) but compares only two faces.

The code is inspired by [Terry Chan's MTCNN Face Recgnition system](https://github.com/Chanstk/FaceRecognition_MTCNN_Facenet)
and [Mandar Joshi's Facenet C++](https://github.com/mndar/facenet_classifier).

The Multi-Task CNN part is from [Li Qi's MTCNN-Light](https://github.com/AlphaQi/MTCNN-light)
but is somewhat modified to compile on Microsoft Visual C++ 2017 Community.

A pre-compiled binary which can be run on Windows (7 or later) without Python is in [the release](https://github.com/brhr-iwao/facenet-compare-cpp/releases).

#### Comparison among Facenet-compare-cpp, [Facenet](https://github.com/davidsandberg/facenet) and [Facematch](https://github.com/arunmandal53/facematch).

L2 distances calculated by using each of facenet-compare-cpp, [compare.py in David Sandberg's facenet](https://github.com/davidsandberg/facenet/blob/master/src/compare.py) and [face_match_demo.py in Arun Mandal's facematch](https://github.com/arunmandal53/facematch/blob/master/face_match_demo.py) with a pretraind model [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) were shown in the following table.   
The test face images are in [David Sandberg's Facenet](https://github.com/davidsandberg/facenet/tree/master/data/images)
or [Arun Mandal's Facematch](https://github.com/arunmandal53/facematch/tree/master/images).

|Face1                   |Face2                   | 	      |L2 distances	|	                 |
|:----------------------|:----------------------|:-------:|:-----------:|:----------------:|
|                        |                        |cpp      |facenet	    |facematch	       |
|Anthony_Hopkins_0001.jpg|Anthony_Hopkins_0002.jpg|0.631916	|0.6283     	|0.5816779         |
| daniel-radcliffe_2.jpg	| daniel-radcliffe_4.jpg	| 0.895936	|0.9134	| 0.94480646	|
| amitab_old.jpg	| amitab_young.jpg	| 0.850119	| 0.8464	| 0.9341758	|
| Barack_Obama.jpg	| IlhamAnas.jpg	| 0.982423	| 1.1558	| 1.239484	|
| dicaprio-the-revenant.jpg	| dicaprio-the-titanic.jpg	| no data(1)	| 0.9046	| 0.9822168	|
| angelina-jolie-young.jpg	| Esha-gupta.jpg	| 1.03429	|1.1811	|1.1570308	|

(1) Unfortunately the facenet-compare-cpp failed to detect face in dicaprio-the-revenant.jpg.   
The results by [face_match_demo.py](https://github.com/arunmandal53/facematch/blob/master/face_match_demo.py) are equivalent to the examples shown in [the article "MTCNN face Detection and Matching using Facenet Tensorflow" by Arun Mandal](https://www.pytorials.com/face-detection-matching-using-facenet).    
Threshold 1.1 for facenet and facematch, 0.95 for facenet-compare-cpp seem to be valid for these examples.
