#pragma once
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

Mat getMedianWaveLenght();

Mat getOrientedField();

Mat normalizedImage();

Mat claheFilter();

Mat orientedField2(Mat& g);

Mat xSignature2(Mat& g, Mat& O);

Mat xSignature3(Mat& g, Mat& O);

Mat medianWavelenght(Mat& xSig);

Mat medianWavelenght2(Mat& xSig);

void printField(string windowName, Mat& O);

void print_xSig(Mat& xSig, Mat& O, Mat& g);


Mat orientedField2_total(Mat& g);

Mat getFrequency();

Mat filter_ridge(Mat& inputImage, Mat& orientationImage, Mat& frequency);

void meshgrid(int kernelSize, cv::Mat& meshX, cv::Mat& meshY);