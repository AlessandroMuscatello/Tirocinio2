#include "tracing.h"
#include "newPoint.h"
#include "improveImage.h"
#include <algorithm>

using namespace std;
using namespace cv;

extern Mat originalImage; //immagine originale
extern Mat O; //matrice delle orientazioni (oriented field)

double beta2 = 0.5; //threshold per la verifica dell'harmonic ratio (DA AUMENTARE?)

//funzione che a partire da un seed (punto p) traccia la linea corrispondente in entrambe le direzioni

void traceLine(Point originalPoint, Mat& finalImage, Mat& finalPath) {
	list<double> waveLenghts; //lista delle larghezze delle ridges calcolate per questa linea
	list<Point> newPoints;
	newPoints.push_front(originalPoint);
	//Mat localLine(originalImage.cols, originalImage.rows, CV_8U);
	//localLine = 0;
	waveLenghts.push_front(5); //primo valore

	Point p = originalPoint;
	Mat omega; //matrice della finestra locale su p
	Mat xSig; //vettore della xSig su p
	Mat fourier; //vettore della trasformata di xSig
	Mat PS; //vettore del Power Spectrum
	bool tracing = true; //condizione di tracciamento
	bool condition; //condizione di OK per il calcolo della xSig dallo spettro filtrato e calcolare il nuovo punto

	while (tracing) {
		double angle = O.at<double>(p.y / 16, p.x / 16) / M_PI * 180; //angolo locale su p
		omega = 255 - getOmega(p, angle, Size(4.5 * waveLenghts.front(), 50)); //oriented window locale con inversione dei colori (neri = valori alti)
		xSig = getNormXSig(omega); //xSignature locale
		PS = getPowerSpectrum(xSig);
		dft(xSig, fourier);
		int n = floor(getFourierMaxIndex(fourier) / 2);
		condition = false;
		cout << "check 1" << endl;
		if (checkHarmonicRatio(PS, n) == false) { //se non va a buon fine viene ricalcolato l'HR con larghezza di omega di 4.5 waveLenght
			omega = 255 - getOmega(p, angle, Size(7.5 * waveLenghts.front(), 50));
			xSig = getNormXSig(omega);
			PS = getPowerSpectrum(xSig);
			dft(xSig, fourier);
			n = ceil(getFourierMaxIndex(fourier) / 2);
			cout << "check 2" << endl;
			if (checkHarmonicRatio(PS, n) == true) //se il ricalcolo va a buon fine 
				condition = true;
		}
		else
			condition = true; //il primo calcolo era andato a buon fine
		cout << "condizione calcolo nuovo punto = " << condition << endl;
		if (condition) { //calcolo la nuova xSig filtrata e il nuovo punto

			//DEBUG
			namedWindow("omega", WINDOW_FREERATIO);
			resizeWindow("omega", Size(250, 100));
			imshow("omega", omega);
			string windowName = "X-Signature";
			CvPlot::Axes axes = CvPlot::plot(xSig, "-");
			CvPlot::Window window(windowName, axes, 480, 640);

			xSig = newXSig(fourier);

			//DEBUG
			string windowName2 = "X-Signature 2";
			CvPlot::Axes axes2 = CvPlot::plot(xSig, "-");
			CvPlot::Window window2(windowName2, axes2, 480, 640);

			int deltaC = getDeltaC(xSig); //spostamento rispetto al centro della finestra del picco
			double newAngle = angle + atan2(deltaC, 0.5 * waveLenghts.front()); //nuovo angolo calcolato
			Point newP(round(p.x + 4 * cos(newAngle)), round( p.y - 4 * sin(newAngle))); //nuovo punto calcolato
			
			newPoints.push_front(newP);
			//line(finalImage, p, newP, Scalar(255, 255, 255), 3); //scrittura sull'immagine finale
			line(finalPath, p, newP, Scalar(0, 0, 0), 1); //scrittura sull'immagine dei path (DEBUG)

			waveLenghts.push_front(omega.size[1] / (n * 2)); //aggiunta della nuova largezza della linea calcolata alla lista
			cout << "angolo calcolato = " << angle << " + " << newAngle - angle << " = " << newAngle << endl;
			cout << "prossima waveLenght calcolata = " << omega.size[1] << " / (" << n << " * 2) = " << waveLenghts.front() << endl;

			imshow("finalImage", finalImage);
			imshow("finalPath", finalPath);
			p = newP;
			cout << "newP = " << newP << endl;
			waitKey(0);
		}
		else {
			cout << "Il punto p = " << p << " è stato scartato";
			drawRidges(finalImage, newPoints, waveLenghts);
			
			break;
		}
		if (finalImage.at<double>(p) == 255) { //se il punto è già stato tracciato interrompi la linea e inserisci una biforcazione
			cout << "biforcazione!" << endl;
			drawRidges(finalImage, newPoints, waveLenghts);
			//system("pause");
			break;
		}
	}
	//INVERTIRE L'ANGOLO
	
	}
		
//funzione che controlla se l'HR soddisfa la condizione di threshold beta2
bool checkHarmonicRatio(Mat& PS, int n) {
	//cout << "controllo harmonic" << endl;
	
	//numero di ridges sul quale è stato calcolato il PS
	cout << "n = " << n << endl;
	if (n - 2 > 0) { //condizione per poter effettuare la valutazione dell'harmonic ratio
		double den = 0; //denominatore per calcolare l'harnomic ratio
		for (int i = 1; i <= n - 2; i++)
			if (den < PS.at<double>(i))
				den = PS.at<double>(i);
		cout << "den = " << den << endl;
		if (den == 0) //non posso calcolare l'harmonic ratio
			return false;

		double HR = PS.at<double>(n) / den; //harmonic ratio
		cout << "HR = " << HR << endl;

		if (HR < beta2) //l'harmonic ratio non soddisfa la condizione di threshold
			return false;
		else
			return true;
	}
	else return false;
}

//funzione che calcola la xSig a partire dal filtraggio dello spettro di fourier
Mat newXSig(Mat& fourier) {
	int maxIndex = getFourierMaxIndex(fourier); //indice del valore massimo nel vettore della trasformata di fourier 

	for (int s = 0; s < fourier.size[0]; s++)
		if ((s < (maxIndex - 1) || s >(maxIndex + 3)) || (s == 2 * maxIndex))
			fourier.at<double>(s) = 0;
	Mat newXSig; //xSig derivata dal filtraggio sul power spectrum
	dft(fourier, newXSig, DFT_INVERSE);
	return newXSig;
}

//funzione che ritorna lo scostamento deltaC dal centro della xSig al picco più vicino
int getDeltaC(Mat& xSig) {
	int cPos = xSig.size[0] / 2 - 1, //indice del centro della xSig
		deltaC = 0; //spostamento rilevato dal centro della xSig al centro della ridge

	while (xSig.at<double>(cPos - 1) < xSig.at<double>(cPos) && xSig.at<double>(cPos) < xSig.at<double>(cPos + 1)) {
		deltaC++;
		cPos += deltaC;
		//cout << "primo if, delta = " << deltaC << ", xSig = " << xSig.at<double>(cPos) << endl;
	}

	while (xSig.at<double>(cPos - 1) > xSig.at<double>(cPos) && xSig.at<double>(cPos) > xSig.at<double>(cPos + 1)){
		deltaC--;
		cPos += deltaC;
		//cout << "secondo if, delta = " << deltaC << ", xSig = " << xSig.at<double>(cPos) << endl;
	}
	return deltaC;
}

//funzione che ritorna la posizione del massimo valore nel vettore della trasformata di fourier
int getFourierMaxIndex(Mat& fourier) {
	int maxIndex = 0; //indice del valore massimo nel vettore della trasformata di fourier 
	double maxValue = -9999;
	for (int i = 1; i < fourier.size[0]; i++) {
		if (fourier.at<double>(i) > maxValue) {
			maxValue = fourier.at<double>(i);
			maxIndex = i;
		}
	}
	return maxIndex;
}

Mat getPowerSpectrum(Mat& xSig) {
	//vedi https://docs.opencv.org/4.5.0/d8/d01/tutorial_discrete_fourier_transform.html per come si calcola il Power Spectrum
	Mat planes[] = { Mat_<double>(xSig.clone()), Mat::zeros(xSig.size(), CV_64F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	return planes[0];
}

void drawRidges(Mat& finalImage, list<Point>& newPoints, list<double>& waveLenghts) {
	while (newPoints.size() > 1) {
		Point first = newPoints.front();
		newPoints.pop_front();
		line(finalImage, first, newPoints.front(), Scalar(255, 255, 255), waveLenghts.front(), 16);
		waveLenghts.pop_front();
	}
}