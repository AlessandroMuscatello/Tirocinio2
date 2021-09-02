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

extern Mat originalImage; //immagine originale (usata solo per le dimensioni)
extern Mat filteredImage;
extern Mat medianLenght;
extern Mat O; //matrice delle orientazioni (oriented field)
double a = 200; //threshold per la selezione del punto
double beta1 = 1; //threshold per il seeding (vedi region quality check)


//funzione che stabilisce se un punto può essere usato come seed per il tracciamento della ridge
bool isSeedingPoint(Point p, Mat& finalImage) {
	if (filteredImage.at<uchar>(p) > a) //PIXEL INTENSITY CHECK
		return false;

	int localLenght = medianLenght.at<double>(p.y / 16, p.x / 16);
	double angle = O.at<double>(p.y / 16, p.x / 16);
	
	if (retracingCheck(p, finalImage, angle, localLenght)) {
		
		if (regionQualityCheck(p, angle, localLenght))
			return true;
		//cout << "region quality NOT pass" << endl;
	}
	
	return false;
}

//funzione per effettuare il retracingCheck
bool retracingCheck(Point p, Mat& finalImage, double angle, int localLenght) {
	Mat omega = getOmega(p, angle, Size(localLenght, localLenght)); //finestra orientata sul punto p dall'immagine filtrata
	Mat omega_ = getOmega_(omega); //finestra orientata binarizzata 
	Mat omegaT = getOmegaT(p, finalImage, angle, localLenght); //finestra orientata sul punto p dalla matrice finale
	multiply(omega_, omegaT, omega); //moltiplica le prime due matrici elemento per elemento e lo scrive sulla terza

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
	
	Mat omega = 255 - getOmega(p, angle, Size(5 * localLenght, localLenght)); //i neri sono le valli e i bianchi sono le ridges
	Mat xSig = getNormXSig(omega); //xSignature della finestra orientata
	
	//vedi https://docs.opencv.org/4.5.0/d8/d01/tutorial_discrete_fourier_transform.html per come si calcola il Power Spectrum
	
	Mat planes[] = { Mat_<double>(xSig.clone()), Mat::zeros(xSig.size(), CV_64F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat PS = planes[0];
	
	Mat fourier;
	dft(xSig, fourier);
	PS.at<double>(0) = 0; // se non si facesse così, 0 è sempre l'indice con il valore più alto;
	double maxVal = 0; 
	int max = 0; //indice del picco massimo di frequenza
	for (int i = 1; i < fourier.size[0]; ++i)
		if (abs(fourier.at<double>(i)) > maxVal) {
			max = i;
			maxVal = fourier.at<double>(i);
		}
	
	/*
	//DEBUG
	string windowName = "Fourier";
	CvPlot::Axes axes = CvPlot::plot(fourier, "-");
	CvPlot::Window window(windowName, axes, 480, 640);

	string windowName3 = "Power Spectrum";
	CvPlot::Axes axes3 = CvPlot::plot(PS, "-");
	CvPlot::Window window3(windowName3, axes3, 480, 640);
	
	string windowName2 = "X-Signature";
	CvPlot::Axes axes2 = CvPlot::plot(xSig, "-");
	CvPlot::Window window2(windowName2, axes2, 480, 640);

	namedWindow("omega", WINDOW_FREERATIO);
	resizeWindow("omega", 250, 100);
	imshow("omega", omega);
	cout << "max = " << max << endl;
	*/
	int n = floor(max / 2);
	//cout << "n = " << n << endl;
	
	if (n - 2 > 0) {
		double den = 0; //denominatore per calcolare l'harnomic ratio
		for (int i = 0; i <= n - 2; i++) //trova il valore massimo che abbia indice minore di max - 2 e lo mette come denominatore per il HR
			if (PS.at<double>(i) > den)
				den = PS.at<double>(i);
		if (den == 0)
			return false;
		
		double num = 0;
		for (int i = n - 1; i <= n + 3; i++) //trova il valore massimo che abbia indice minore di max - 2 e lo mette come denominatore per il HR
			if (PS.at<double>(i) > den)
				num = PS.at<double>(i);


		double HR = num / den; //harmonic ratio
		//cout << "HR = " << num << " / " << den << " = " << HR << endl;
		//waitKey(0);
		
		if (HR > beta1) 
			return true;
	}
	
	return false;
}

//funzione per ottenere la finestra orientata sul punto p dall'immagine filtrata
Mat getOmega(Point p, double angle, Size windowSize) {
	float tx, ty; //valori per la traslazione
	Mat translated_image; //matrice traslata che deve essere ruotata
	Mat omega; //finestra orientata
	Point center(filteredImage.cols / 2, filteredImage.rows / 2);
	
	//valori per la traslazione
	tx = center.x - p.x;
	ty = center.y - p.y;
	float warp_values[] = { 1, 0, tx, 0, 1, ty }; //valori per la matrice di traslazione
	Mat translation_matrix = Mat(2, 3, CV_32F, warp_values); //matrice di traslazione
	warpAffine(filteredImage, translated_image, translation_matrix, filteredImage.size()); //effettuo la traslazione dell'immagine
	
	Mat rotation_matix = getRotationMatrix2D(center, - angle / M_PI * 180 + 90, 1.0); //matrice di rotazione
	warpAffine(translated_image, omega, rotation_matix, translated_image.size()); //effettuo la rotazione dell'immagine

	Range rowRange(center.y - windowSize.height / 2, center.y + windowSize.height / 2 + 1),
		  colRange(center.x - windowSize.width  / 2, center.x + windowSize.width  / 2 + 1); //range per il taglio (il +1 alla fine è perchè l'ultima riga / colonna non viene inclusa quindi va aggiunta a mano)
	omega(rowRange, colRange).copyTo(omega); //effettua il taglio
	
											 /*
	//DEBUG
	cout << "rowRange = " << rowRange << endl;
	cout << "colrange = " << colRange << endl;
	circle(translated_image, Point(translated_image.cols / 2, translated_image.rows / 2), 2, Scalar(255, 255, 0));

	namedWindow("translated", WINDOW_FREERATIO);
	imshow("translated", translated_image);
	resizeWindow("translated", 500, 500);

	namedWindow("omega", WINDOW_FREERATIO);
	imshow("omega", omega);
	resizeWindow("omega", 500, 500);
	
	cout << omega <<endl;
	waitKey(0);
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
	Point center(filteredImage.cols / 2, filteredImage.rows / 2);

	//valori per la traslazione
	tx = center.x - p.x;
	ty = center.y - p.y;
	float warp_values[] = { 1, 0, tx, 0, 1, ty }; //valori per la matrice di traslazione
	Mat translation_matrix = Mat(2, 3, CV_32F, warp_values); //matrice di traslazione

	warpAffine(finalImage, translated_image, translation_matrix, finalImage.size()); //effettuo la traslazione dell'immagine
	Mat rotation_matix = getRotationMatrix2D(center, -angle * 180 / M_PI + 90, 1.0); //matrice di rotazione

	//cout << "locallenght = " << localLenght << endl;
	warpAffine(translated_image, omega, rotation_matix, translated_image.size()); //effettuo la rotazione dell'immagine
	Range rowRange(center.y - localLenght / 2, center.y + localLenght / 2 + 1),
		colRange(center.x - localLenght / 2, center.x + localLenght / 2 + 1);
	omega(rowRange, colRange).copyTo(omega); //effettua il taglio in base alla dimensione locale della ridge
	omega.convertTo(omega, CV_8U);
	return omega;
}

//funzione per la xSignature normalizzata
Mat getNormXSig(Mat& omega) {
	Mat xSig; //matrice della xSig da calcolare
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
	xSig /= (max - min); //normalizza xSig in modo da avere valori in [0, 1]
	
	/*
	//DEBUG
	cout << "norm xSig" << endl << xSig << endl;
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