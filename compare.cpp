/*
// compare.cpp
//
// Calculates a distance between two faces
// by using a davidsandberg/facenet pre-trained model.
// https://github.com/davidsandberg/facenet
//
// This code is inspired by Terry Chan's MTCNN Face Recgnition system:
// https://github.com/Chanstk/FaceRecognition_MTCNN_Facenet
// and Mandar Joshi's Facenet C++ Classifiler:
// https://github.com/mndar/facenet_classifier
//
*/

// Includes and related sources are taken from Li Qi's MTCNN
// https://github.com/AlphaQi/MTCNN-light
// They were modified by brhr-iwao
// to avoid the error "ACCESS_MASK ambiguous symbol" on MSVC.
#include "network.h" 
#include "mtcnn.h"

#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"

#define HEIGHT 160
#define WIDTH  160
#define DEPTH  3

using namespace std;

// Read a facenet model
bool getSession(string graph_path, unique_ptr<tensorflow::Session> &session)
{
	tensorflow::GraphDef graph_def;

	if(!tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def).ok())
	{
		cout << "Read proto ERROR" << endl;
		return false;
	}

	else cout << "Read a proto..." << endl;

	tensorflow::SessionOptions sess_opt;

	(&session)->reset(tensorflow::NewSession(sess_opt));

	if(!session->Create(graph_def).ok())
	{
		cout << "Create graph ERROR" << endl;
		return false;
	}

	else cout << "Create a graph..." << endl;

	return true;

}

// Find and crop a face from an image
bool getFace(cv::Mat image, cv::Mat& face)
{
	const int margin = 11;
	// const int margin = 0;

	if(image.data == NULL )
	{
		cout << "getFace function ERROR, The input cv::Mat is empty" << endl;
		return false;
	}

	cv::mtcnn find(image.cols, image.rows);
	find.findFace(image);

	for(vector<struct Bbox>::iterator it = find.thirdBbox_.begin() ; 
		it != find.thirdBbox_.end() ; it++)
	{
		if( (*it).exist )
		{
			cv::Rect faceRect((*it).y1 - margin, 
				      (*it).x1 - margin, 
				      (*it).y2 - (*it).y1 + margin,
				      (*it).x2 - (*it).x1 + margin );
			face = image(faceRect).clone(); // first found face only
			return true; 
		}
	}

	cout << "getFace function ERROR, no face found." << endl;
	return false; // no face found.
}

// Project cv::Mat to Tensorflow::Tensor
void getImageTensor(tensorflow::Tensor& input_tensor, cv::Mat& image)
{
	cv::resize(image,image, cv::Size( HEIGHT, WIDTH ));
	auto input_tensor_mapped = input_tensor.tensor<float,4>();

	// Converts "WIDTH x HEIGHT x 3" to "WIDTH x 3*HEIGHT x 1"
	cv::Mat gray = image.reshape(1, image.rows*3);

	// mean and standard deviation of pixels in image
	cv::Mat mean;
	cv::Mat stddev;
	cv::meanStdDev( gray, mean, stddev );

	//double mean_pxl = mean.at<double>(0);
	double stddev_pxl = stddev.at<double>(0);

	// prewhiten
	image.convertTo(image, CV_64FC1);
	image = image - mean;
	image = image/stddev_pxl;

	// copying the data into the corresponding tensor
	for( int y = 0; y < HEIGHT ; ++y)
	{
		const double* src_row = image.ptr<double>(y);

		for( int x = 0 ; x < WIDTH ; ++x)
		{
			const double* src_pxl = src_row + (DEPTH * x);

			for( int c = 0 ; c < DEPTH ; ++c )
			{
				const double* src_val = src_pxl + 2-c;
				// const double* src_val = src_pxl + c;
				input_tensor_mapped(0,y,x,c) = *src_val;
			}
		}
	}
}

// Embed a face tensor to a 128-demensions vector
bool embed(const unique_ptr<tensorflow::Session> &session, tensorflow::Tensor &image, float* facevec)
{
	tensorflow::Tensor phase_train( tensorflow::DT_BOOL, tensorflow::TensorShape() );

	phase_train.scalar<bool>()() = false;

	vector<tensorflow::Tensor> outputs;

	tensorflow::Status run_status = session->Run({{"input:0", image},{"phase_train:0", phase_train}},
													{"embeddings:0"},
													{},
													&outputs);
	if(!run_status.ok())
	{
		cout << "Running model faild" << run_status << endl;
		return false;
	}

	else cout << "Running the model..." << endl;

	float* p = outputs[0].flat<float>().data();

	for (int i = 0; i < 128; i++)
		facevec[i] = p[i];

	return true;

}

int main(int argc, char* argv[] )
{
	if(argc < 4 )
	{
		cout << "comp.exe" << endl;
		cout << "Description: Calculates a distance between two faces" << endl;
		cout << "             by using a davidsandberg/facenet pre-trained model." << endl;
		cout << "usage: comp proto_name image1_name image2_name" << endl;
		cout << "e.g. comp 20170512-110547.pb Anthony_Hopkins_0001.jpg Anthony_Hopkins_0002.jpg" << endl;
		cout << "Note: The image1 and image2 must contain only one face." << endl;
		cout << "Please download a proto (pre-trained model) file from https://github.com/davidsandberg/facenet" << endl;

		return 0;
	}

	string graph_path = argv[1];
	cv::Mat image1mat = cv::imread(argv[2]);

    if( image1mat.data == NULL )
	{
		cout << "Image " << argv[2] << " loading is faild." << endl;
		cout << "This image may be too large to load." << endl;
		return 0;
	}

	cv::Mat image2mat = cv::imread(argv[3]);

	if( image2mat.data == NULL )
	{
		cout << "Image " << argv[3] << " loading is faild." << endl;
		cout << "This image may be too large to load." << endl;
		return 0;
	}

	cv::Mat face1mat;
	cv::Mat face2mat;

	tensorflow::Tensor face1tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, HEIGHT, WIDTH, DEPTH}));
	tensorflow::Tensor face2tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, HEIGHT, WIDTH, DEPTH}));

	unique_ptr<tensorflow::Session> session;

	float* face1vec = new float[128];
	float* face2vec = new float[128];

	getSession(graph_path, session);

	getFace(image1mat, face1mat);
	getImageTensor(face1tensor, face1mat);
	embed(session, face1tensor, face1vec);

	getFace(image2mat, face2mat);
	getImageTensor(face2tensor, face2mat);
	embed(session, face2tensor, face2vec);

	double sum = 0;

	for(int i = 0; i<128 ; i++)
		sum += ( face1vec[i] - face2vec[i] )*( face1vec[i] - face2vec[i] );

	cout << "Euclidean distance between " << argv[2] << " and " << argv[3] << " is " << sqrt(sum) << endl;

	session->Close();

	delete face1vec;
	delete face2vec;

	return 1;
}

