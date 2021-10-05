#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <list>

//libreria cvplot per grafici
#define CVPLOT_HEADER_ONLY
#include <CvPlot/core.h>
#include <CvPlot/cvplot.h>

void traceLine(cv::Point p, cv::Mat& finalImage, cv::Mat& finalPath);

bool checkHarmonicRatio(cv::Mat& fourier, int maxFourierIndex);

cv::Mat newXSig(cv::Mat& fourier);

cv::Mat newXSig2(cv::Mat& fourier, int n);

int getDeltaC(cv::Mat& xSig);

int getFourierMaxIndex(cv::Mat& fourier);

int getN(cv::Mat& xSig);

cv::Mat getPowerSpectrum(cv::Mat& xSig);

void drawRidges(cv::Mat& finalImage, std::list<cv::Point>& newPoints, std::list<double>& waveLenghts);