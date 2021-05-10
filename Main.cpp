#include <iostream>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cv.h>
#include <highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

void method_one(void);
void method_two(void);

int main(int argc, char** argv) {
	
	string method_1 = "ORB";
	string method_2 = "BRISK";
	string input;
	
	std::cout << "TYPE IN A FEATURE DETECTION METHOD WITH CAPS LOCK" << std::endl;
	std::cout << "ORB FOR ORB METHOD OR BRISK FOR BRISK METHOD :  " << std::endl;
	std::getline (std::cin, input);
	std::cout << "*******************************************************************************" << std::endl;

	if (method_1.compare(input) == 0) {
		method_one();
	}if (method_2.compare(input) == 0) {
		method_two();
	}

	std::cout << "THANK YOU DR. FRAZER NOBLE" << std::endl;
	std::cout << "PRESS ESC TO EXIT | THEN PRESS F5 TO EXCECUTE THIS AGAIN" << std::endl;
	
	waitKey(0);
	return 0;
}

void method_one() {
	Mat img = imread("image_1.jpg");
	Mat out;

	std::vector<KeyPoint> kp;
	
	int nfeatures = 500;
	float scale_factor = 1.2f;
	int nlevels = 8;
	int edge_threshold = 15; 
	int first_level = 0;
	int wta_k = 2;
	int score_type = ORB::HARRIS_SCORE;
	int patch_size = 31;
	int fast_threshold = 20;

	Ptr<ORB> detector = ORB::create(
		nfeatures,
		scale_factor,
		nlevels,
		edge_threshold,
		first_level,
		wta_k,
		score_type,
		patch_size,
		fast_threshold
	);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	detector->detect(img, kp);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	float duration = duration_cast <microseconds>(t2 - t1).count();
	float duration_milliseconds = duration / 1000;
	float duration_seconds = duration / 1000000;

	std::cout << "POINT LOCATIONS ARE FOLLOWS" << std::endl;
	if (kp.size() > 0) {
		for (int i = 0; i <= kp.size(); i++) {
			float x0 = kp[i].pt.x;
			float y0 = kp[i].pt.y;
			std::cout << "POINT NUMBER " << i << " X-POSITION = " << x0 << "  " << i << " Y-POSITION = " << y0 << std::endl;
		}
	}
	std::cout << "*******************************************************************************" << std::endl;
	std::cout << "ORB FOUND " << kp.size() << " KEYPOINTS" << std::endl;
	std::cout << "TIME TAKEN: " << duration << "  MICROSECONDS" << std::endl;
	std::cout << fixed;
	std::cout.precision(2);
	std::cout << "TIME TAKEN: " << duration_milliseconds << "  MILLISECONDS" << std::endl;
	std::cout << "TIME TAKEN: " << duration_seconds << "  SECONDS" << std::endl;

	drawKeypoints(img, kp, out, Scalar::all(255));
	
	imshow("KEYPOINTS", out);
	imwrite("ORB_OUTPUT.jpg", out);

	return;
}

void method_two() {
	Mat img = imread("image_1.jpg");
	Mat out;

	std::vector<KeyPoint> kp;

	int thresh = 30;
	int octaves = 3;
	float pattern_scale = 1.0f;

	Ptr<BRISK> detector = BRISK::create(
		thresh,
		octaves,
		pattern_scale
	);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	detector->detect(img, kp);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	float duration = duration_cast <microseconds>(t2 - t1).count();
	float duration_milliseconds = duration / 1000;
	float duration_seconds = duration / 1000000;

	std::cout << "POINT LOCATIONS ARE FOLLOWS" << std::endl;
	if (kp.size() > 0) {
		for (int i = 0; i <= kp.size(); i++) {
			float x0 = kp[i].pt.x;
			float y0 = kp[i].pt.y;
			std::cout << "POINT NUMBER " << i << " X-POSITION = " << x0 << "  " << i << " Y-POSITION = " << y0 << std::endl;
		}
	}
	std::cout << "**********************************************************************************" << std::endl;
	std::cout << "BRISK FOUND " << kp.size() << " KEYPOINTS" << std::endl;
	std::cout << "TIME TAKEN: " << duration << "  MICROSECONDS" << std::endl;
	std::cout << fixed;
	std::cout.precision(2);
	std::cout << "TIME TAKEN: " << duration_milliseconds << "  MILLISECONDS" << std::endl;
	std::cout << "TIME TAKEN: " << duration_seconds << "  SECONDS" << std::endl;

	drawKeypoints(img, kp, out, Scalar::all(255));

	imshow("KEYPOINTS", out);
	imwrite("BRISK_OUTPUT.jpg", out);

	return;
}