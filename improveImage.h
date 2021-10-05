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

Mat xSignature4(Mat& g, Mat& O);

Mat getOrientedWindow(Point p, Mat& g, double angle, Size windowSize);

Mat medianWavelenght(Mat& xSig);

Mat medianWavelenght2(Mat& xSig);

Mat medianWavelenght3(Mat& xSig);

void printField(string windowName, Mat& O);

void print_xSig(Mat& xSig, Mat& O, Mat& g);