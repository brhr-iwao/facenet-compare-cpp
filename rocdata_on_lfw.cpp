/*
// rocdata_on_lfw.cpp
//
// Calculates a roc_data (data for ROC curve calculation)
// from a LFW-style face dataset by using
// one of davidsandberg/facenet pre-trained model.
// https://github.com/davidsandberg/facenet
//
// This code is inspired by Terry Chan's MTCNN Face Recgnition system:
// https://github.com/Chanstk/FaceRecognition_MTCNN_Facenet
// and Mandar Joshi's Facenet C++ Classifiler:
// https://github.com/mndar/facenet_classifier
//
//
//  "pairs.txt" must be in the directoy which contains the executable.
//
//  "pairs.txt" is a list of pairs of images
//   as same as pairs.txt(http://vis-www.cs.umass.edu/lfw/pairs.txt) of LFW.
//
//	Usage: rocdata_on_lfw [graph_path] [protobuf_name] [image_path] [extension_name_with_period]
//  e.g. rocdata_on_lfw models 20180402-114759.pb mtcnnpy_160 .png
//
*/

#include <math.h>
#include <string>
#include <iostream>
#include <fstream>

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
		cout << "Reading protobuf ERROR" << endl;
		return false;
	}

	else cout << "Read the protobuf ..." << endl;

	tensorflow::SessionOptions sess_opt;

	(&session)->reset(tensorflow::NewSession(sess_opt));

	if(!session->Create(graph_def).ok())
	{
		cout << "Creating graph ERROR" << endl;
		return false;
	}

	else cout << "Create a graph..." << endl;

	return true;

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

double distanceL2( string graph_path, string graph_name, char* imageFile1, char* imageFile2 )
{
	cv::Mat image1mat = cv::imread(imageFile1);

	if (image1mat.data == NULL)
	{
		cerr << "Image " << imageFile1 << " loading faild." << endl;
		return -1.0;
	}

	cv::Mat image2mat = cv::imread(imageFile2);

	if (image2mat.data == NULL)
	{
		cerr << "Image " << imageFile2 << " loading faild." << endl;
		return -1.0;
	}

	tensorflow::Tensor face1tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, HEIGHT, WIDTH, DEPTH }));
	tensorflow::Tensor face2tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, HEIGHT, WIDTH, DEPTH }));

	unique_ptr<tensorflow::Session> session;

	float* face1vec = new float[128];
	float* face2vec = new float[128];

	cout << endl << "Session launched for " << imageFile1 << " and " << imageFile2 << endl;

	string path_graph = graph_path + '/' + graph_name;
	getSession(path_graph, session);

	getImageTensor(face1tensor, image1mat);
    embed(session, face1tensor, face1vec);

    getImageTensor(face2tensor, image2mat);
	embed(session, face2tensor, face2vec);

	double sum = 0.0;

	for (int i = 0; i < 128; i++)
			sum += (face1vec[i] - face2vec[i])*(face1vec[i] - face2vec[i]);

	session->Close();
	cout << "Session finished" << endl;

	return sqrt(sum);

}

bool generateRocData( string graph_path,
	                  string graph_name,
	                  const char* pairs_file, 
	                  const char* dirName, const char* extName,
	                  const char* out_rocdata_file )
{
	ifstream rs( pairs_file, ios::in );
	ofstream ws( out_rocdata_file, ios::out );

	if (rs.fail())
	{
		cerr << "Can't open " << pairs_file << endl;
		return false;
	}

	if (ws.fail())
	{
		cerr << "Can't open " << out_rocdata_file << endl;
		return false;
	}

	static char line[128];

	// Skip the first line and initialize the rs
	rs.getline(line, sizeof(line));

	while ( true )
	{
		char fileName1[128] = "";
		char fileName2[128] = "";

		rs.getline(line, sizeof(line));

		if (rs.eof()) break;

		// Split the line to the four(or three) tokens
		static char* p;
		char p1[64], p2[8], p3[64], p4[8];
		char delimit[] = " \t"; // Space and tab are the specified for strtok.

		p = strtok(line, delimit);
		if (p != NULL) strcpy(p1, p);
		else strcpy(p1, "NULL");

		p = strtok(NULL, delimit);
		if (p != NULL) strcpy(p2, p);
		else strcpy(p2, "NULL");

		p = strtok(NULL, delimit);
		if (p != NULL) strcpy(p3, p);
		else strcpy(p3, "NULL");

		p = strtok(NULL, delimit);
		if (p != NULL) strcpy(p4, p);
		else strcpy(p4, "NULL");

		// Same identity pair
		if (strcmp(p4, "NULL") == 0)
		{
            // fileNmme1
			strcat(fileName1, dirName);
			strcat(fileName1, "/");
			strcat(fileName1, p1);
			strcat(fileName1, "/");
			strcat(fileName1, p1);
			strcat(fileName1, "_");

			int num;

			num = atoi(p2);
			sprintf(p2, "%04d", num);

			strcat(fileName1, p2);
			strcat(fileName1, extName);

			// fileName2
			strcat(fileName2, dirName);
			strcat(fileName2, "/");
			strcat(fileName2, p1);
			strcat(fileName2, "/");
			strcat(fileName2, p1);
			strcat(fileName2, "_");

			num = atoi(p3);
			sprintf(p3, "%04d", num);

			strcat(fileName2, p3);
			strcat(fileName2, extName);

			//double L2;

			double L2 = distanceL2(graph_path, graph_name, fileName1, fileName2);

			if(L2 >= 0.0)
				ws << showpoint << L2 << noshowpoint << "   " << "0" << endl << flush;
			else ws << "X.XXX    0 " << endl << flush;
		
		}

		// Different identity pair
		else if (strcmp(p4, "NULL") != 0)
		{
			// fileName1
			strcat(fileName1, dirName);
			strcat(fileName1, "/");
			strcat(fileName1, p1);
			strcat(fileName1, "/");
			strcat(fileName1, p1);
			strcat(fileName1, "_");

			int num;

			num = atoi(p2);
			sprintf(p2, "%04d", num);

			strcat(fileName1, p2);
			strcat(fileName1, extName);

			// fileName2
			strcat(fileName2, dirName);
			strcat(fileName2, "/");
			strcat(fileName2, p3);
			strcat(fileName2, "/");
			strcat(fileName2, p3);
			strcat(fileName2, "_");

			num = atoi(p4);
			sprintf(p4, "%04d", num);

			strcat(fileName2, p4);
			strcat(fileName2, extName);

			double L2 = distanceL2(graph_path, graph_name, fileName1, fileName2);

			if (L2 >= 0.0)
				ws << showpoint << L2 << noshowpoint << "   " << "1" << endl << flush;
			else ws << "X.XXX    1 " << endl << flush;
		
		}

		else
		{
			cerr << "Error during parsing a line.";
			break;
		}

	}

	rs.close();
	ws.close();

	return true;

}

int main(int argc, char* argv[] )
{
	if(argc < 5 )
	{
		cout << "rocdata_on_lfw.exe" << endl;
		cout << "Description: Generate Roc data text file (rocdata.txt) from LFW dataset" << endl;
		cout << "             by using a davidsandberg/facenet pre-trained model." << endl;
		cout << "The Roc data text file is formatted as:" << endl;
		cout << "The 1st column: L2 distance between a pair of faces" << endl;
		cout << "The 2nd column: 0 for a same identity pair or 1 for a different identity pair" << endl;
		cout << "Usage: rocdata_on_lfw [graph_path] [protobuf_name] [image_path] [extension_name_with_period]" << endl;
		cout << "e.g. rocdata_on_lfw models 20180402-114759.pb mtcnnpy_160 .png" << endl;
		cout << "" << endl;
		cout << "Please download a protobuf file from https://github.com/davidsandberg/facenet before use." << endl;

		return 0;
	}

	generateRocData(argv[1], argv[2], "pairs.txt", argv[3], argv[4], "roc_data.txt");

	cout << "roc_data.txt generation done" << endl << "Press ENTER to EXIT...";
	getchar();
	return 1;
}

