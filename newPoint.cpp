#include "newPoint.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

//libreria cvplot per grafici
#define CVPLOT_HEADER_ONLY
#include <CvPlot/core.h>
#include <CvPlot/cvplot.h>


using namespace std;
using namespace cv;

extern Mat originalImage; //immagine originale
double a = 200; //threshold per la selezione del punto
double beta1 = 1; //threshold per il seeding (vedi region quality check)


//funzione che stabilisce se un punto può essere usato come seed per il tracciamento della ridge
bool isSeedingPoint(Point p, Mat& img, Mat& finalImage, Mat& medianLenght, Mat& O) {
	originalImage = img;
	if (originalImage.at<uchar>(p.y, p.x) > a) //PIXEL INTENSITY CHECK
		return 0;

	int localLenght = medianLenght.at<double>(p.y / 16, p.x / 16);
	double angle = O.at<double>(p.y / 16, p.x / 16) / M_PI * 180;
	
	if (retracingCheck(p, finalImage, angle, localLenght)) {
		if (regionQualityCheck(p, angle, localLenght))
			return true;
	}
	
	return false;
}

//funzione per effettuare il retracingCheck
bool retracingCheck(Point p, Mat& finalImage, double angle, int localLenght) {
	Mat omega = getOmega(p, angle, Size(localLenght, localLenght)); //finestra orientata sul punto p
	Mat omega_ = getOmega_(omega); //finestra orientata binarizzata
	Mat omegaT = getOmegaT(p, finalImage, angle, localLenght); //finestra orientata sul punto p dalla matrice finale

	multiply(omega_, omegaT, omega);

	int T = 0;
	for (auto it = omega.begin<uchar>(), end = omega.end<uchar>(); it != end; ++it)
		T += *it;
	
	if (T > 0) //se T > 0 il punto si trova su una ridge già tracciata
		return false;
	else 
		return true;
}

//funzione che controlla la regione adiacente al punto per stabilire se si trova su una ridge non 
bool regionQualityCheck(Point p, double angle, int localLenght) {
	Mat omega = 255 - getOmega(p, angle, Size(2.5 * localLenght, localLenght)); //i neri sono le valli e i bianchi sono le ridges

	Mat xSig = getNormXSig(omega); //xSignature della finestra orientata

	Mat fourier; //matrice del power spectrum 
	dft(xSig, fourier);
	fourier.at<double>(0) = 0; 

	double maxVal = 0; 
	int max = 0; //indice del picco di frequenza
	for (int i = 0; i < fourier.size[0]; ++i)
		if (fourier.at<double>(i) > maxVal) {
			max = i;
			maxVal = fourier.at<double>(i);
		}
	cout << "max = " << max << endl;

	if (max - 2 > 0) {
		
		double den = 0; //indice del denominatore per calcolare l'harnomic ratio
		cout << "fourier = " << endl << fourier << endl;
		for (int i = 0; i < max - 2; i++) {
			cout << "i = "<< i << endl;
			if (den < fourier.at<double>(i))
				den = fourier.at<double>(i);
		}
		
		if (den == 0)
			return false;

		double HR = fourier.at<double>(max) / den; //harmonic ratio

		cout << "HR = " << fourier.at<double>(max) << " / " << fourier.at<double>(den) << " = " << HR << endl;
		if (HR > beta1) 
			return true;
	}
	
	return false;
}

//funzione per ottenere la finestra orientata sul punto p
Mat getOmega(Point p, double angle, Size windowSize) {
	float tx, ty; //valori per la traslazione
	Mat translated_image; //matrice traslata che deve essere ruotata
	Mat omega;

	//valori per la traslazione
	tx = originalImage.cols / 2 - p.x;
	ty = originalImage.rows / 2 - p.y;
	float warp_values[] = { 1, 0, tx, 0, 1, ty }; //valori per la matrice di traslazione
	Mat translation_matrix = Mat(2, 3, CV_32F, warp_values); //matrice di traslazione

	warpAffine(originalImage, translated_image, translation_matrix, originalImage.size()); //effettuo la traslazione dell'immagine
	Mat rotation_matix = getRotationMatrix2D(Point(originalImage.cols / 2, originalImage.rows / 2), - angle + 90, 1.0); //matrice di rotazione
	
	//cout << "locallenght = " << localLenght << endl;
	warpAffine(translated_image, omega, rotation_matix, translated_image.size()); //effettuo la rotazione dell'immagine
	Rect centerCut(
		Point(omega.cols / 2 - windowSize.width / 2, omega.rows / 2 - windowSize.height / 2),
		Point(omega.cols / 2 + windowSize.width / 2, omega.rows / 2 + windowSize.height / 2)); //rettangolo per il taglio al centro
	omega(centerCut).copyTo(omega); //effettua il taglio in base alla dimensione locale della ridge
	
	/* //DEBUG
	circle(translated_image, Point(translated_image.cols / 2, translated_image.rows / 2), 2, Scalar(255, 255, 0));

	namedWindow("translated", WINDOW_FREERATIO);
	imshow("translated", translated_image);
	resizeWindow("translated", 500, 500);

	namedWindow("omega", WINDOW_FREERATIO);
	imshow("omega", omega);
	resizeWindow("omega", 500, 500);
	
	cout << omega <<endl;
	*/

	return omega;
}

//funzione per la binarizzazione di omega secondo la media
Mat getOmega_(Mat& omega) {
	Mat omega_; //matrice binarizzata secondo la media
	omega.copyTo(omega_);
	double sum = 0;
	for (auto it = omega_.begin<uchar>(), end = omega_.end<uchar>(); it != end; ++it) {
		sum += *it;
	}
	double mean = sum / omega.total();

	for (auto it = omega_.begin<uchar>(), end = omega_.end<uchar>(); it != end; ++it) //binarizzazione di omega in base al valore medio
		if (*it < mean)
			*it = 1;
		else
			*it = 255;
	return omega_;
}

//funzione per ottenere la finestra orientata dall'immagine finale
Mat getOmegaT(Point p, Mat finalImage, double angle, int localLenght) {
	float tx, ty; //valori per la traslazione
	Mat translated_image; //matrice traslata che deve essere ruotata
	Mat omega;

	//valori per la traslazione
	tx = finalImage.cols / 2 - p.x;
	ty = finalImage.rows / 2 - p.y;
	float warp_values[] = { 1, 0, tx, 0, 1, ty }; //valori per la matrice di traslazione
	Mat translation_matrix = Mat(2, 3, CV_32F, warp_values); //matrice di traslazione

	warpAffine(finalImage, translated_image, translation_matrix, finalImage.size()); //effettuo la traslazione dell'immagine
	Mat rotation_matix = getRotationMatrix2D(Point(finalImage.cols / 2, finalImage.rows / 2), -angle + 90, 1.0); //matrice di rotazione

	//cout << "locallenght = " << localLenght << endl;
	warpAffine(translated_image, omega, rotation_matix, translated_image.size()); //effettuo la rotazione dell'immagine
	Rect centerCut(
		Point(omega.cols / 2 - localLenght / 2, omega.rows / 2 - localLenght / 2),
		Point(omega.cols / 2 + localLenght / 2, omega.rows / 2 + localLenght / 2)); //rettangolo per il taglio al centro
	omega(centerCut).copyTo(omega); //effettua il taglio in base alla dimensione locale della ridge
	omega.convertTo(omega, CV_8U);
	return omega;
}

//funzione per la xSignature normalizzata
Mat getNormXSig(Mat& omega) {
	Mat xSig;
	double sum, val, min = 256, max = 0;

	for (int i = 0; i < omega.cols; ++i) { //calcolo la xSig
		sum = 0;
		for (int j = 0; j < omega.rows; j++) {
			sum += omega.at<uchar>(j, i);
		}
		val = sum / omega.rows;
		xSig.push_back(val);
		if (val < min)
			min = val;
		if (val > max)
			max = val;
	}
	
	xSig = xSig - min;
	xSig /= (max - min);
	
	
	//cout << "norm xSig" << endl << xSig << endl;

	/*
	//DEBUG
	namedWindow("omega", WINDOW_FREERATIO);
	imshow("omega", omega);
	resizeWindow("omega", 250, 100);

	string windowName = "xsig";
	CvPlot::Axes axes = CvPlot::plot(xSig, "-");
	CvPlot::Window window(windowName, axes, 480, 640);

	waitKey(0);
	*/

	return xSig;
}