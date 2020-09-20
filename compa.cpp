/*
// compa.cpp
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
// MTCNN Face Detection:
// https://github.com/OAID/FaceDetection
//
// usage: compa --gp=graph_path --mn=model_name --i1=image1_name --i2=image2_name -s image_size -m margin
// e.g. compa --gp=models --mn=20180402-114759.pb --i1=Anthony_Hopkins_0001.jpg --i2=Anthony_Hopkins_0002.jpg -s 160 -m 32
// 
// Note: The image1 and image2 must contain only one face.
// Please download a protobuf file from https://github.com/davidsandberg/facenet
//
*/


#include "mtcnn.hpp" 
#include "utils.hpp"

#include <math.h>
#include <stdlib.h>

#include "getopt.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"


#define DEPTH  3

using namespace std;

// Read a facenet model
bool getSession(string path_graph, unique_ptr<tensorflow::Session> &session)
{
	tensorflow::GraphDef graph_def;

	cout << endl;

	if(!tensorflow::ReadBinaryProto(tensorflow::Env::Default(), path_graph, &graph_def).ok())
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
bool getFace(cv::Mat image, cv::Mat& face, string graph_path, int margin)
{
#ifndef MIN
#define MIN(a,b)  ((a)<(b)?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b)  ((a)>(b)?(a):(b))
#endif


	if(image.data == NULL )
	{
		cout << "getFace function ERROR, The input cv::Mat is empty" << endl;
		return false;
	}

	vector<face_box> face_info;

	mtcnn* p_mtcnn = mtcnn_factory::create_detector("tensorflow");

	if (p_mtcnn == nullptr)
	{
		cerr << "Face detector with tensorflow does not work." << endl;
		return false;
	}

	p_mtcnn->load_model(graph_path);
	p_mtcnn->detect(image, face_info);

	if (!face_info.empty())
	{
		face_box& box = face_info[0]; // first found face only

		int x0 = MAX(box.x0 - margin / 2, 0);
		int y0 = MAX(box.y0 - margin / 2, 0);
		int x1 = MIN(box.x1 + margin / 2, image.cols);
		int y1 = MIN(box.y1 + margin / 2, image.rows);

		cv::Rect faceRect(x0, y0, x1 - x0, y1 - y0);

		face = image(faceRect).clone();

		delete p_mtcnn;
		return true;
	}


	cerr << "getFace function ERROR, no face found." << endl;
	delete p_mtcnn;
	return false; // no face found.
}

// Project cv::Mat to Tensorflow::Tensor
void getImageTensor(tensorflow::Tensor& input_tensor, cv::Mat& image, int size)
{
	cv::resize(image, image, cv::Size(size, size));
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
	// for( int y = 0; y < HEIGHT ; ++y)
	for (int y = 0; y < size; ++y)
	{
		const double* src_row = image.ptr<double>(y);

		// for( int x = 0 ; x < WIDTH ; ++x)
		for (int x = 0; x < size; ++x)
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
	if(argc < 5 )
	{
		cout << endl;
		cout << "compa.exe" << endl;
		cout << "Description: Calculates a distance between two faces" << endl;
		cout << "             by using a davidsandberg/facenet pre-trained model." << endl;
		cout << "usage: compa --gp=graph_path --mn=model_name --i1=image1_name --i2=image2_name -s image_size -m margin" << endl;
		cout << "e.g. compa --gp=models --mn=20180402-114759.pb --i1=Anthony_Hopkins_0001.jpg --i2=Anthony_Hopkins_0002.jpg -s 160 -m 32" << endl;
		cout << "Note: The image1 and image2 must contain only one face." << endl;
		cout << "Please download a protobuf file from https://github.com/davidsandberg/facenet" << endl;

		return 1;
	}

	char gp[MAX_PATH] = "";
	char mn[128] = "";
	char i1[MAX_PATH] = "";
	char i2[MAX_PATH] = "";
	int size = 160;
	int margin = 32;

	struct option longopts[] =
	{
		{"gp", required_argument, NULL, 'g'},
		{"mn", required_argument, NULL, 'n'},
		{"i1", required_argument, NULL, 'i'},
		{"i2", required_argument, NULL, 'j'},
		{"image_size", required_argument, NULL, 's'},
		{"margin", required_argument, NULL, 'm'},
		{0,0,0,0}
	};

	int opt = 0;

	while (( opt = getopt_long(argc, argv, "g:n:i:j:s:m:", longopts, NULL) ) != -1)
	{
		switch(opt)
		{
			case 'g':
				strcpy(gp, optarg);
				break;
			case 'n':
				strcpy(mn, optarg);
				break;
			case 'i':
				strcpy(i1, optarg);
				break;
			case 'j':
				strcpy(i2, optarg);
				break;
			case 's':
				size = atoi(optarg);
				break;
			case 'm':
				margin = atoi(optarg);
				break;
			default:
				break;
		}
	}

	string graph_path(gp);
	string graph_name(mn);
	string path_graph = graph_path + '/' + graph_name;
	cv::Mat image1mat = cv::imread(i1);

    if( image1mat.data == NULL )
	{
		cout << "Image " << i1 << " loading is faild." << endl;
		cout << "This image may be not specified or too large to load." << endl;
		return 0;
	}

	cv::Mat image2mat = cv::imread(i2);

	if( image2mat.data == NULL )
	{
		cout << "Image " << i2 << " loading is faild." << endl;
		cout << "This image may be not specified or too large to load." << endl;
		return 0;
	}

	cv::Mat face1mat;
	cv::Mat face2mat;

	tensorflow::Tensor face1tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, size, size, DEPTH }));
	tensorflow::Tensor face2tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, size, size, DEPTH }));

	unique_ptr<tensorflow::Session> session;

	float* face1vec = new float[128];
	float* face2vec = new float[128];

	getSession(path_graph, session);

	getFace(image1mat, face1mat, graph_path, margin);
	getImageTensor(face1tensor, face1mat, size);
	embed(session, face1tensor, face1vec);

	getFace(image2mat, face2mat, graph_path, margin);
	getImageTensor(face2tensor, face2mat, size);
	embed(session, face2tensor, face2vec);

	double sum = 0;

	for(int i = 0; i<128 ; i++)
		sum += ( face1vec[i] - face2vec[i] )*( face1vec[i] - face2vec[i] );

	cout << "Euclidean distance between " << i1 << " and " << i2 << " is " << sqrt(sum) << endl;

	session->Close();

	delete face1vec;
	delete face2vec;

	return 0;
}

