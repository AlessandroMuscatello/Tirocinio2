#include "tracing.h"
#include "newPoint.h"
#include "improveImage.h"
#include <algorithm>

using namespace std;
using namespace cv;

extern Mat originalImage; //immagine originale
extern Mat O; //matrice delle orientazioni (oriented field)
extern Mat medianLenght;

double beta2 = 0.3; //threshold per la verifica dell'harmonic ratio (DA AUMENTARE?)

//funzione che a partire da un seed point traccia la linea corrispondente in entrambe le direzioni
void traceLine(Point seed, Mat& finalImage, Mat& finalPath) {
	list<double> waveLenghts; //lista delle larghezze delle ridges calcolate per questa linea
	list<Point> newPoints, bifurcations, terminations;
	newPoints.push_front(seed);
	//Mat localLine(originalImage.cols, originalImage.rows, CV_8U);
	//localLine = 0;
	//waveLenghts.push_front(medianLenght.at<double>(Point(seed.x / 16, seed.y / 16))); //primo valore
	waveLenghts.push_front(5);
	for (double i = 0; i <= M_PI; i += M_PI) { //CALCOLA IN UNA DIREZIONE E POI LA INVERTE
		cout << "i = " << i << endl;
		Point p = seed;
		double angle = O.at<double>(p.y / 16, p.x / 16) + i; //angolo locale su p
		Mat omega; //matrice della finestra locale
		Mat xSig; //vettore della xSig di omega
		Mat fourier; //vettore della trasformata di xSig
		Mat PS; //vettore del Power Spectrum		
		bool condition; //condizione di OK per il calcolo della xSig dallo spettro filtrato e calcolare il nuovo punto

		while (true) {
			Point testP(round(p.x + 4 * cos(angle)), round(p.y - 4 * sin(angle)));
			omega = 255 - getOmega(testP, angle, Size(2.5 * waveLenghts.front(), waveLenghts.front())); //oriented window locale con inversione dei colori (ridge = valori alti)
			xSig = getNormXSig(omega); //xSignature locale
			PS = getPowerSpectrum(xSig); //PS locale
			dft(xSig, fourier);
			int n = getN(xSig); // numero di ridges nella finestra omega
			condition = false;

			/*
			namedWindow("omega", WINDOW_FREERATIO);
			resizeWindow("omega", Size(omega.size[0], omega.size[1]));
			imshow("omega", omega);
			waitKey(0);
			cout << "check 1" << endl;
			*/

			if (checkHarmonicRatio(PS, n) == false) { //se non va a buon fine viene ricalcolato l'HR con larghezza di omega aumentata
				omega = 255 - getOmega(testP, angle, Size(4.5 * waveLenghts.front(), waveLenghts.front()));
				xSig = getNormXSig(omega);
				PS = getPowerSpectrum(xSig);
				dft(xSig, fourier);
				n = getN(xSig);

				/*
				namedWindow("omega", WINDOW_FREERATIO);
				resizeWindow("omega", Size(omega.size[0], omega.size[1]));
				imshow("omega", omega);
				//n = ceil(getFourierMaxIndex(fourier) / 2); 
				//cout << "n == " << n << endl;
				cout << "check 2" << endl;
				*/

				if (checkHarmonicRatio(PS, n) == true) //se il ricalcolo va a buon fine 
					condition = true;
			}
			else
				condition = true; //il primo calcolo era andato a buon fine
			//cout << "condizione calcolo nuovo punto = " << condition << endl;
			condition = true;
			if (condition) { //calcolo la nuova xSig filtrata e il nuovo punto
				
				
				//DEBUG
				namedWindow("omega", WINDOW_FREERATIO);
				resizeWindow("omega", Size(omega.size[0], omega.size[1]));
				imshow("omega", omega);
				cout << "numero ridge n = " << n << endl;
				
				CvPlot::Axes axes = CvPlot::plot(xSig, "-");
				CvPlot::Window window("xsig", axes, 480, 640);
				
				CvPlot::Axes axes3 = CvPlot::plot(PS, "-");
				CvPlot::Window window3("PS", axes3, 480, 640);
				

				//xSig = newXSig(fourier);
				xSig = newXSig2(fourier, n);

				/*
				//DEBUG 
				CvPlot::Axes axes2 = CvPlot::plot(xSig, "-");
				CvPlot::Window window2("xsig filtrata", axes2, 480, 640);
				*/
				

				int deltaC = getDeltaC(xSig); //spostamento rispetto al centro della finestra del picco
				double newAngle = angle - atan2(deltaC, 0.5 * waveLenghts.front()) * 0.5; //nuovo angolo calcolato
				Point newP(round(p.x + 4 * cos(newAngle)), round(p.y - 4 * sin(newAngle))); //nuovo punto calcolato
				newPoints.push_front(newP);
				line(finalPath, p, newP, Scalar(0), 1); //scrittura sull'immagine dei path
				finalImage.at<double>(p) = 255;
				waveLenghts.push_front(omega.size[1] / (n * 1.5)); //aggiunta della nuova largezza della linea calcolata alla lista
				p = newP;
				angle = newAngle;
				
				/*
				//DEBUG 
				cout << "deltaC = " << deltaC << endl;
				cout << "angolo calcolato = " << angle << " + " << newAngle - angle << " = " << newAngle << endl;
				cout << "newP = " << newP << endl;
				cout << "prossima waveLenght calcolata = " << omega.size[1] << " / (" << n << " * 2) = " << waveLenghts.front() << endl;
				imshow("finalImage", finalImage);
				imshow("finalPath", finalPath);
				waitKey(0);
				*/
				
			}
			else { //il punto non passa i due test dell'HR
				//cout << "scarto p = " << p << endl;
				drawRidges(finalImage, newPoints, waveLenghts);
				terminations.push_back(p);
				break;
			}
			if (finalImage.at<double>(p) == 255) { //se il punto è già stato tracciato interrompi la linea e inserisci una biforcazione
				//cout << "biforcazione in p = " << p << endl;
				drawRidges(finalImage, newPoints, waveLenghts);
				bifurcations.push_back(p);
				//system("pause");
				break;
			}
		}

	}


}
		
//funzione che controlla se l'HR soddisfa la condizione di threshold beta2
bool checkHarmonicRatio(Mat& PS, int n) {
	
	//cout << "n = " << n << endl;
	if (n - 2 > 0) { //condizione per poter effettuare la valutazione dell'harmonic ratio
		double den = 0; //denominatore per calcolare l'harnomic ratio
		for (int i = 1; i <= n - 2; i++)
			if (den < PS.at<double>(i))
				den = PS.at<double>(i);
		//cout << "den = " << den << endl;
		if (den == 0) //non posso calcolare l'harmonic ratio
			return false;

		double num = 0; //numeratore per calcolare l'harmonic ratio
		for (int i = n - 1; i <= n + 3; i++) //trova il valore massimo che abbia indice compreso tra n-1 e n+3 e lo mette come numeratore per il HR
			if (PS.at<double>(i) > num)
				num = PS.at<double>(i);

		double HR = num / den; //harmonic ratio
		//cout << "HR = " << HR << endl;

		if (HR < beta2) //l'harmonic ratio non soddisfa la condizione di threshold
			return false;
		else
			return true;
	}
	else return false;
}

//funzione che calcola la xSig a partire dal filtraggio dello spettro di fourier (NON USATA)
Mat newXSig(Mat& fourier) {
	int maxIndex = getFourierMaxIndex(fourier); //indice del valore massimo nel vettore della trasformata di fourier 

	for (int s = 0; s < fourier.size[0]; s++)
		if ((s < (maxIndex - 1) || s >(maxIndex + 3)) || (s == 2 * maxIndex))
			fourier.at<double>(s) = 0;
	Mat newXSig; //xSig derivata dal filtraggio sul power spectrum
	dft(fourier, newXSig, DFT_INVERSE);
	return newXSig;
}

//funzione che calcola la xSig a partire dal filtraggio dello spettro di fourier
Mat newXSig2(Mat& fourier, int n) {

	for (int s = 0; s < fourier.size[0]; s++)
		if ((s < (n - 1) || s >(n + 3)))
			fourier.at<double>(s) = 0;
	Mat newXSig; //xSig derivata dal filtraggio sul power spectrum
	idft(fourier, newXSig);
	return newXSig;
}

//funzione che ritorna lo scostamento deltaC dal centro della xSig al picco più vicino
int getDeltaC(Mat& xSig) {
	int cPos = xSig.size[0] / 2 , //indice del centro della xSig
		deltaC = 0; //spostamento rilevato dal centro della xSig al centro della ridge
	//cout << "xSig = " << endl << xSig << endl;
	//cout << "xsig(cPos) = xsig(" << cPos << ") =" << xSig.at<double>(cPos) << endl;
	
	while (xSig.at<double>(cPos - 1) < xSig.at<double>(cPos) && xSig.at<double>(cPos) < xSig.at<double>(cPos + 1)) {
		deltaC++;
		cPos++;
		//cout << "primo if, delta = " << deltaC << ", xSig = " << xSig.at<double>(cPos) << endl;
	}

	while (xSig.at<double>(cPos - 1) > xSig.at<double>(cPos) && xSig.at<double>(cPos) > xSig.at<double>(cPos + 1)){
		deltaC--;
		cPos++;
		//cout << "secondo if, delta = " << deltaC << ", xSig = " << xSig.at<double>(cPos) << endl;
	}
	return deltaC;
}

//funzione che ritorna l'indice del massimo valore nel vettore di fourier
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

//funzione che ritorna il numero di ridge nella xSig (derivate)
int getN(Mat& xSig) {
	Mat der;
	double xSigMean = mean(xSig)[0]; 
	int n = 0;
	Sobel(xSig, der, xSig.depth(), 0, 1, 1);
	for (int i = 1; i < der.size[0] - 2; i++) {
		if (der.at<double>(i) > 0 && der.at<double>(i + 1) <= 0 && xSig.at<double>(i) > xSigMean)
			n++;
	}
	/*
	cout << "n == " << n << endl;
	CvPlot::Axes axes = CvPlot::plot(xSig, "-");
	CvPlot::Window window("xsig", axes, 480, 640);

	CvPlot::Axes axes2 = CvPlot::plot(der, "-");
	CvPlot::Window window2("derivate", axes2, 480, 640);
	*/

	return n;
}

//funzione che permette di estrarre il Power Spectrum dalla matrice di xSignature
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

//funzione che stampa sulla finalImage le ridges rilevate con la loro larghezza
void drawRidges(Mat& finalImage, list<Point>& newPoints, list<double>& waveLenghts) {
	while (newPoints.size() > 1) {
		Point first = newPoints.front();
		newPoints.pop_front();
		line(finalImage, first, newPoints.front(), Scalar(255), waveLenghts.front() , 16);
		waveLenghts.pop_front();
	}
}