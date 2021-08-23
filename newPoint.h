#pragma once
#include <opencv2/core.hpp>
using namespace cv;

bool isSeedingPoint(Point p, Mat& image, Mat& finalImage, Mat& medianLenght, Mat& O);

bool retracingCheck(Point p, Mat& finalImage, double angle, int localLenght);

bool regionQualityCheck(Point p, double angle, int localLenght);

Mat getOmega(Point p, double angle, Size windowSize);

Mat getOmega_(Mat& omega);

Mat getOmegaT(Point p, Mat finalImage, double angle, int localLenght);

Mat getNormXSig(Mat& omega);