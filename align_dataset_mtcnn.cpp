
/*
//  align_dataset_mtcnn.cpp
//
//   Align images in a face dataset.
//   The dataset must be configured as same as LFW(http://vis-www.cs.umass.edu/lfw/) in advance.
//   ( i.e. Image file names are Firstname_Familyname_0001.jpg, FirstName_Familyname_0002.jpg,... and so on.
//     They are put in the directory named Firstname_Familyname)
//
//   Run this program such as:
//   align_dataset_mtcnn -i input_Directory -o output_Directory -s output_image_size -m output_image_margin e output_imege_file_extention
//   e.g. align_dataset_mtcnn -i lfw -o mtcnn_160 -s 160 -m 32 e png
//
// Calculates a distance between two faces
// by using a davidsandberg/facenet pre-trained model.
// https://github.com/davidsandberg/facenet
//
// This code is inspired by Terry Chan's MTCNN Face Recgnition system:
// https://github.com/Chanstk/FaceRecognition_MTCNN_Facenet
// and Mandar Joshi's Facenet C++ Classifiler:
// https://github.com/mndar/facenet_classifier
// MTCNN Face Detection:
// https://github.com/OAID/FaceDetection
//
*/

#include <string>
#include <fstream>
#include <time.h>

#include "mtcnn.hpp"
#include "utils.hpp"
// Opencv includes is not required.

#include <getopt.h>
#include <dirent.h>

#ifdef WIN32
#include <direct.h>
#endif
#include <sys/stat.h>

#ifndef MAX_PATH
    #define MAX_PATH 260
#endif

char inputDir[MAX_PATH];
char outputDir[MAX_PATH];

bool lfwStyleDirName(DIR* dirStream, char* parentDir, char* dirName, char* dirContainsImage)
{
	if ((dirStream == NULL) || (parentDir == NULL)) return false;

	struct dirent* ent;

	if ((ent = readdir(dirStream)) != NULL)
	{
		if (strcmp(ent->d_name, ".") &&
			strcmp(ent->d_name, "..") &&
			(ent->d_type == DT_DIR))
		{
			char innerDirName[256] = "";
			strcpy(innerDirName, parentDir);
			strcat(innerDirName, "/");
			strcat(innerDirName, ent->d_name);

			strcpy(dirName, innerDirName);
			strcpy(dirContainsImage, ent->d_name);
			return true;
		}

		else return false;
	}

	else return false;
}

bool lfwStyleFileName(DIR* d, char* parentDir, char* fileName, char* basename,std::string& withoutExt)
{
	if ((d == NULL) || (parentDir == NULL)) return false;

	struct dirent* e;

	if ((e = readdir(d)) != NULL)
	{
		if (e->d_type == DT_REG) // Regular file case
		{
			char fullPathName[256] = "";
			strcpy(fullPathName, parentDir);
			strcat(fullPathName, "/");
			strcat(fullPathName, e->d_name);

			std::string s(e->d_name);
			s.erase(s.find_last_of("."), std::string::npos);
			withoutExt = s;

			strcpy(fileName, fullPathName);
			strcpy(basename, e->d_name);
			return true;
		}

		else return false;
	}

	else return false;
}

bool cropImage(char* in, char* out, int s, int m, int& x0, int& y0, int& x1, int& y1)
{
#ifndef MIN
#define MIN(a,b)  ((a)<(b)?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b)  ((a)>(b)?(a):(b))
#endif

	x0 = 0;
	y0 = 0;
	x1 = 0;
	y1 = 0;

	std::string inputName(in);
	std::string outputName(out);

	cv::Mat inputMat = cv::imread(inputName);

	if (!inputMat.data)
	{
		std::cerr << "failed to read image file: " << in << std::endl;
		x0 = -1;
		y0 = -1;
		x1 = -1;
		y1 = -1;
		return false;
	}

	std::string model_dir = "models";
	std::vector<face_box> face_info;

	mtcnn * p_mtcnn = mtcnn_factory::create_detector("tensorflow");

	if (p_mtcnn == nullptr)
	{
		std::cerr << "Face detector with tensorflow does not work." << std::endl;
		x0 = -1;
		y0 = -1;
		x1 = -1;
		y1 = -1;
		return false;
	}

	p_mtcnn->load_model(model_dir);
	p_mtcnn->detect(inputMat, face_info);

	if (!face_info.empty())
	{
		face_box& box = face_info[0]; // first found face only

		x0 = MAX(box.x0 - m / 2, 0);
		y0 = MAX(box.y0 - m / 2, 0);
		x1 = MIN(box.x1 + m / 2, inputMat.cols);
		y1 = MIN(box.y1 + m / 2, inputMat.rows);

		cv::Rect faceRect(x0, y0, x1-x0, y1-y0);

		cv::Mat face = inputMat(faceRect).clone();

		cv::resize(face, face, cv::Size(s, s), s / face.cols, s / face.rows);

		cv::imwrite(outputName, face);

		delete p_mtcnn;

		return true;
	}

	else
	x0 = -1;
	y0 = -1;
	x1 = -1;
	y1 = -1;
	return false;
}

int main( int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cout << std::endl;
		std::cout << "Run this program such as :" << std::endl;
		std::cout << "align_dataset_mtcnn -i input_Directory -o output_Directory -s output_image_size -m output_image_margin e output_imege_file_extention" << std::endl;
		std::cout << "e.g. align_dataset_mtcnn -i lfw -o mtcnn_160 -s 160 -m 32 e png" << std::endl;
		return 1;
	}

	// initialize with defaut values
	int size = 160;
	int margin = 32;
	char extension[8] = "png";

	int opt;
	const char* optstring = "i:o:s:m:e:";

	while ((opt = getopt(argc, argv, optstring)) != -1)
	{
		switch (opt)
		{
		case 'i':
			strcpy(inputDir, optarg);
			break;
		case 'o':
			strcpy(outputDir, optarg);
			break;

		case 's':
			size = atoi(optarg);
			break;

		case 'm':
			margin = atoi(optarg);
			break;

		case 'e':
			strcpy(extension, optarg);
			break;

		default:
			break;
		}

	}

#ifdef WIN32
	struct _stat s;
	if (_stat(inputDir, &s) == -1)
	{
		std::cerr << inputDir << " is not found!" << std::endl;
		return 1;
	}
#else
	struct stat s;
	if (stat(inputDir, &s) == -1)
	{
		std::cerr << inputDir << " is not found!" << std::endl;
		return 1;
	}
#endif


	char directoryName[256] = "";
	char fn[256] = "";

	char dci[128] = "";
	char bn[128] = "";

	DIR* lfwDirStream;
	DIR* dirStream;

#ifdef WIN32
	//struct _stat s;
	if (_stat(outputDir, &s) == -1)
		_mkdir(outputDir);
#else
	//struct stat s;
	if (stat(outputDir, &s) == -1)
		mkdir(outputDir);
#endif

	// create log
	char logName[256] = "";
	strcpy(logName, outputDir);

	char logBaseName[256] = "";
	struct tm *t;
	time_t tt;
	time(&tt);
	t = localtime(&tt);
	sprintf(logBaseName, "bounding_box_%d%02d%02d%02d%02d%02d.txt",
		t->tm_year - 100,
		t->tm_mon + 1,
		t->tm_mday,
		t->tm_hour,
		t->tm_min,
		t->tm_sec);

	strcat(logName, "/");
	strcat(logName, logBaseName);

	std::ofstream ofs(logName);
	// end create log

	lfwDirStream = opendir(inputDir);

	// The two lines are neccessary for Windows only..?
#ifdef WIN32
	lfwStyleDirName(lfwDirStream, inputDir, directoryName, dci); // skip .
	lfwStyleDirName(lfwDirStream, inputDir, directoryName, dci); // skip ..
#endif

	while (lfwStyleDirName(lfwDirStream, inputDir, directoryName, dci))
	{

		dirStream = opendir(directoryName);

		std::string s;

#ifdef WIN32
		lfwStyleFileName(dirStream, directoryName, fn, bn, s);// skip .
		lfwStyleFileName(dirStream, directoryName, fn, bn, s);// skip ..
#endif

		while (lfwStyleFileName(dirStream, directoryName, fn, bn, s))
		{
			char ofn[256] ="";

			strcat(ofn, outputDir);
		    strcat(ofn, "/");
			strcat(ofn, dci);

#ifdef WIN32
			struct _stat t;
			if (_stat(ofn, &t) == -1)
				_mkdir(ofn);
#else
			struct stat t;
			if (stat(ofn, &t) == -1)
				mkdir(ofn);
#endif

			strcat(ofn, "/");
			//strcat(ofn, bn);
			//printf("%s\n", ofn);
			strcat(ofn, s.c_str());
			strcat(ofn, ".");
			strcat(ofn, extension);

			std::cout << "Aligning " << ofn << "..." << std::endl;

			int tl_x, tl_y, br_x, br_y;

			cropImage(fn, ofn, size, margin, tl_x, tl_y, br_x, br_y);
			ofs << ofn << " " << tl_x << "  " << tl_y << " " << br_x << " " << br_y << std::endl << std::flush;

		}

		closedir(dirStream);

	}

	closedir(lfwDirStream);

	return 0;


}