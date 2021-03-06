#include "improveImage.h"
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


using namespace std;
using namespace cv;

extern Mat originalImage;

int dim = 16; //dimensione della finestra per il calcolo della direzione

Mat getMedianWaveLenght() {
	
	Mat g = normalizedImage(); //immagine normalizzata (CV_8UC1)
	cout << "calcolata la normImage" << endl;

	Mat O = orientedField2(g); //matrice di orientazione delle ridges (il valore dell'orientazione ? nel centro dei blocchi 16x16) (CV_64F)
	cout << "calcolato l'orientedField" << endl;

	Mat xSig = xSignature3(g, O); //matrice che contiene tutte le xSignature per ogni blocco 16x16. Dimensioni della matrice = [g.rows / dim, g.cols / dim, 32] (CV_64F)
	cout << "calcolato la xSignature " << endl;

	Mat medianLenght = medianWavelenght2(xSig);
	cout << "calcolato la medianLenght " << endl;

	/*
	//DEBUG
	imshow("g", g);	

	printField("O", O);

	print_xSig(xSig, O, g); //DEBUG
	waitKey(0);
	*/
	

	return medianLenght;
}

//funzione che recupera l'Oriented field
Mat getOrientedField() {
	Mat g = normalizedImage();
	return orientedField2(g);
}

//funzione che calcola l'immagine normalizzata (utilizzata per la stima della frequenza delle ridges)
Mat normalizedImage()
{
	double var;//varianza dell'immagine originale
	double mean; //valore della media di grigio dell'immagine 

	double m0 = 100; //coefficiente di normalizzazione
	double var0 = 1000; //coefficiente di normalizzazione

	Mat g(originalImage.rows, originalImage.cols, CV_8UC1); //immagine normalizzata

	int imagePixels = originalImage.rows * originalImage.cols; //numero di pixel dell'immagine originale
	
	double _mean = 0; //var di appoggio per la media del valore di grigio
	for (auto it = originalImage.begin<uchar>(), end = originalImage.end<uchar>(); it != end; ++it) //calcolo della media di grigio
		_mean += (double)(*it);
	mean = _mean / imagePixels;

	
	double _var = 0; //var di appoggio per la varianza
	for (auto it = originalImage.begin<uchar>(), end = originalImage.end<uchar>(); it != end; ++it) //calcolo della varianza
		_var += pow(*it - mean, 2);
	var = _var / imagePixels;

	//cout <<"var ="<< var << endl;

	
	for (auto it = originalImage.begin<uchar>(), end = originalImage.end<uchar>(), gIt = g.begin<uchar>(); it != end; ++it, ++gIt) {
		if ((double)(*it) > mean) // sensibilit? di lettura del grigio
			*gIt = m0 + sqrt(var0 * pow(*it - mean, 2) / var); //chiari in uscita
		else
			*gIt = m0 - sqrt(var0 * pow(*it - mean, 2) / var); //scuri in uscita
	}
	
	return g;
}

//restituisce l'immagine originale filtrata dall'algoritmo CLAHE
Mat claheFilter() {
	Ptr<CLAHE> c = createCLAHE();
	Mat filteredImage;
	c[0].apply(originalImage, filteredImage);
	//imshow("CLAHE", filteredImage);
	//waitKey(0);
	return filteredImage;
}

//funzione che calcola le stime delle direzioni delle ridges in blocchi 16x16 pixel
//@param g = l'immagine normalizzata sulla quale calcolare i gradienti
Mat orientedField2(Mat& g) {
	
	Mat sigma(originalImage.rows / dim, originalImage.cols / dim, CV_64F); //matrice della stima dell'orientazione (non continua)
	Mat filter = getGaussianKernel(5, -1, CV_64F) * getGaussianKernel(5, -1, CV_64F).t(); //maschera gaussiana per l'applicazione del filtro passa basso
	Mat O(sigma.rows, sigma.cols, CV_64F); //effettiva orientazione (campo continuo)

	//calcolo le derivate prime dell'immagine e le metto dentro le matrici
	Mat dx, dy; //derivate prime dell'immagine lungo le direzioni
	Mat gFloat;
	g.convertTo(gFloat, CV_64F);
	Sobel(gFloat, dx, gFloat.depth(), 1, 0); //Sobel esegue la derivata lungo una direzione dell'immagine
	Sobel(gFloat, dy, gFloat.depth(), 0, 1);

	for (int i = 0; i < sigma.rows; i++) {
		for (int j = 0; j < sigma.cols; j++) {
			Rect block(j * dim, i * dim, dim, dim); //ATTENZIONE LE RIGHE E LE COLONNE SONO COORDINATE INVERTITE RISPETTO AD X E Y

			Mat subDx = dx(block), //sottomatrici dei gradienti create a partire dal blocco 16x16
				subDy = dy(block);
			float gxx = 0, gyy = 0, gxy = 0; //variabili d'appoggio per la stima delle direzioni
			//cout << "(i, j) = (" << i << ", " << j << ") -> block = {(" << block.tl() << "), (" << block.br() << ")}" << endl;
			
			for (int u = 0; u < dim; u++) {
				for (int v = 0; v < dim; v++) {
					gxy += subDx.at<double>(u, v) * subDy.at<double>(u, v);
					gxx += pow(subDx.at<double>(u, v), 2);
					gyy += pow(subDy.at<double>(u, v), 2);

					//cout << "gxy, gxx, gyy = " << gxy << " " << gxx << " " << gyy << endl;

					//cout << "vx = 2 * " << (int)subDx.at<uchar>(u, v) << " * " << (int)subDy.at<uchar>(u, v) << " = " << vx << ";  vy = " << pow(subDx.at<uchar>(u, v), 2) << " - " << pow(subDy.at<uchar>(u, v), 2) << ") = " << vy << endl;
				}
			}

			sigma.at<double>(i, j) = 0.5 * atan2(2 * gxy, - gxx + gyy) ; //scrittura della stima dell'orientazione nella matrice sigma

			//cout << "sigma("<<i<<", "<<j<<") = " << sigma.at<double>(i, j) << endl;
		}
	}

	//cout << "sigma = " << endl << sigma << endl;

	//ora ? necessario convertire il campo sigma in un campo vettoriale continuo. Per farlo si applica un filtro che "appiattisce" le curve
	Mat fi_x(sigma.rows, sigma.cols, CV_64F), //componente x del campo vettoriale continuo
		fi_y(sigma.rows, sigma.cols, CV_64F); //componente y del campo vettoriale continuo
	Mat fi_x_(sigma.rows, sigma.cols, CV_64F), //componente x del campo vettoriale continuo con filtro passa basso applicato
		fi_y_(sigma.rows, sigma.cols, CV_64F); //componente y del campo vettoriale continuo con filtro passa basso applicato

	for (int i = 0; i < sigma.rows; i++) {
		for (int j = 0; j < sigma.cols; j++) {
			fi_x.at<double>(i, j) = cos(2 * sigma.at<double>(i, j));
			fi_y.at<double>(i, j) = sin(2 * sigma.at<double>(i, j));
			//cout << fi_x.at<double>(i, j) << " " << sigma.at<double>(i, j) << endl;
		}
	}
	//vengono copiati i valori di sigma per il contorno, poi quelli interni verranno modificati dal ciclo seguente 
	sigma.copyTo(O);

	for (int i = 2; i + 2 < O.rows; i++) {
		for (int j = 2; j + 2 < O.cols; j++) {
			fi_x_.at<double>(i, j) = 0;
			fi_y_.at<double>(i, j) = 0;
			for (int u = -2; u < 3; u++) {
				for (int v = -2; v < 3; v++) {
					//cout << "(u, v) = " << u << ", " << v << endl;
					fi_x_.at<double>(i, j) += filter.at<double>(u + 2, v + 2) * fi_x.at<double>(i + u, j + v);
					fi_y_.at<double>(i, j) += filter.at<double>(u + 2, v + 2) * fi_y.at<double>(i + u, j + v);
				}
			}
			O.at<double>(i, j) = 0.5 * atan2(fi_y_.at<double>(i, j), fi_x_.at<double>(i, j));
			//cout << "o(" << i << ", " << j << ") = " << O.at<double>(i, j) << "   sigma(i, j) = " << sigma.at<double>(i, j) << endl;

		}
	}
	//printField("sigma", sigma);
	//cout << "sigma = " << endl << sigma << endl;
	//waitKey(0);
	return O;
}

/*funzione che calcola la xSignature per ogni blocco 16x16 che sia sufficientemente lontano dal bordo dell'immagine (NON USATA)
* @param g = immagine normalizzata
* @param O = matrice di orientazione
*/
Mat xSignature2(Mat& g, Mat& O) {
	int size[] = { g.rows / dim, g.cols / dim, 32 }; //dimensione della matrice della xSignature (prima e seconda dimensione uguali a quelle di O)
	Mat xSig(3, size, CV_64F); //matrice della xSignature (dimensione (g.rows / 16) x (g.cols/ 16) x 32)
	xSig = 0;	
	/*
	Mat out(g.size[0], g.size[1], CV_8U);
	out = 0;
	namedWindow("points", WINDOW_FREERATIO);
	*/
	
	for (int i = 2; i < xSig.size[0] - 2; i++) { // i e j sono uguali a 2 per non leggere anche fuori dal bordo
		for (int j = 2; j < xSig.size[1] - 2; j++) {
			for (int k = 0; k < xSig.size[2]; k++) {
				for (int v = -7; v < 9; v++) {
					double x = (i * 16) + (k - 15.0) * cos(O.at<double>(i, j) + M_PI_2) + v * sin(O.at<double>(i, j) + M_PI_2);
					double y = (j * 16) + (k - 15.0) * sin(O.at<double>(i, j) + M_PI_2) - v * cos(O.at<double>(i, j) + M_PI_2);
					/*
					//cout << "(i, j) = (" << i << ", " << j << ")  g(x, y) = g(" << x << " " << y << ") = "<< g.at<uchar>(x, y) << endl; 
					out.at<uchar>(x, y) = g.at<uchar>(x, y); 
					imshow("points", out); 
					waitKey(1);
					*/
					
 					xSig.at<double>(i, j, k) += g.at<uchar>(x, y);
				}
				xSig.at<double>(i, j, k) /= 16;
				//cout << "xSig (" << i << ", " << j << ", " << k << ") = " << xSig.at<double>(i, j, k) << endl;
			}	
		}
	}
	return xSig;
}

/*signature creata con una linea e l'iteratore della linea
@param g = immagine normalizzata
@param O = matrice di orientazione
*/
Mat xSignature3(Mat& g, Mat& O) {
	int size[] = { g.rows / dim, g.cols / dim, 32 }; //dimensione della matrice della xSignature (prima e seconda dimensione uguali a quelle di O)
	Mat xSig(3, size, CV_64F); //matrice della xSignature (dimensione (g.rows / 16) x (g.cols/ 16) x 32)
	xSig = 0;
	
	/*
	//DEBUG
	Mat out(g.size(), CV_8U);
	out = 0;
	namedWindow("xsig", WINDOW_FREERATIO);
	*/
	
	for (int i = 2; i < xSig.size[0] - 2; i++) { // i e j sono uguali a 2 per non leggere anche fuori dal bordo
		for (int j = 2; j < xSig.size[1] - 2; j++) {
			//out = 255; //debug
			for (int k = 0; k < xSig.size[2]; k++) {
				
				double yBegin = (i * 16) + (k - 15.0) * cos(O.at<double>(i, j) ) + (-9) * sin(O.at<double>(i, j) );
				double xBegin = (j * 16) + (k - 15.0) * sin(O.at<double>(i, j) ) - (-9) * cos(O.at<double>(i, j) );
				double yEnd = (i * 16) + (k - 15.0) * cos(O.at<double>(i, j) ) + 9 * sin(O.at<double>(i, j) );
				double xEnd = (j * 16) + (k - 15.0) * sin(O.at<double>(i, j) ) - 9 * cos(O.at<double>(i, j) );

				Point begin(xBegin, yBegin), //creo i punti di inizio e fine sui quali iterare
						end(xEnd, yEnd);
				LineIterator it(g, begin, end); //creo l'iteratore che preleva tutti i punti tra begin e end

				//cout << "(begin, end) = (" << begin << ", " << end << ")  it.count = " <<it.count<< endl;
				for (int v = 0; v < it.count; v++, ++it){
					xSig.at<double>(i, j, k) +=  g.at<uchar>(it.pos());
					
					/*
					//DEBUG
					//cout << "v = " << v <<"  p(x, y) = "<< it.pos() << endl;
					out.at<uchar>(it.pos()) = originalImage.at<uchar>(it.pos());
					imshow("xsig", out);
					*/
					
				}
				//waitKey(1); //debug
				if (it.count)
					xSig.at<double>(i, j, k) /= it.count;
				else
					xSig.at<double>(i, j, k) = 0;
			}
			/*
			imwrite("outLineIterator.bmp", out);
			waitKey(0);//debug
			*/
		}
	}
	return xSig;
}

/*signature creata da una oriented window derivata dalla roto - traslazione della matrice g(NON USATA)
@param g = immagine normalizzata
@param O = matrice di orientazione
*/
Mat xSignature4(Mat& g, Mat& O) {
	int size[] = { g.rows / dim, g.cols / dim, 32 }; //dimensione della matrice della xSignature (prima e seconda dimensione uguali a quelle di O)
	Mat xSig(3, size, CV_64F); //matrice della xSignature (dimensione (g.rows / 16) x (g.cols/ 16) x 32)
	xSig = 0;

	for (int i = 2; i < xSig.size[0] - 2; i++) { // i e j sono uguali a 2 per non leggere anche fuori dal bordo
		for (int j = 2; j < xSig.size[1] - 2; j++) {
			Point center(j * 16 + 7, i * 16 + 7); //centro della finestra 16x16
			double angle = O.at<double>(i, j);
			Mat orientedWindow = getOrientedWindow(center, g, angle, Size(32, 16));
			Mat v;
			
			for (int k = 0; k < 32; k++) {
				for (int v = 0; v < 16; v++) {
					xSig.at<double>(i, j, k) += orientedWindow.at<uchar>(v, k);
				}
				xSig.at<double>(i, j, k) /= 16;
				v.push_back(xSig.at<double>(i, j, k));
			}
			/*
			//DEBUG
			CvPlot::Axes axes = CvPlot::plot(v, "-");
			CvPlot::Window window("xSig", axes, 480, 640);
			namedWindow("oriented", WINDOW_FREERATIO);
			imshow("oriented", orientedWindow);
			imshow("g", g);
			waitKey(0);
			
			*/
		}
	}
	return xSig;
}

//funzione per ottenere una finestra orientata di centro p
Mat getOrientedWindow(Point p, Mat& g, double angle, Size windowSize) {
	float tx, ty; //valori per la traslazione
	Mat translated_image; //matrice traslata che deve essere ruotata
	Mat omega; //finestra orientata
	Point center(g.cols / 2, g.rows / 2); //centro della finestra

	//valori per la traslazione
	tx = center.x - p.x;
	ty = center.y - p.y;
	float warp_values[] = { 1, 0, tx, 0, 1, ty }; //valori per la matrice di traslazione
	Mat translation_matrix = Mat(2, 3, CV_32F, warp_values); //matrice di traslazione
	warpAffine(g, translated_image, translation_matrix, g.size()); //effettuo la traslazione dell'immagine

	Mat rotation_matix = getRotationMatrix2D(center, (-angle / M_PI * 180) + 90, 1.0); //matrice di rotazione
	warpAffine(translated_image, omega, rotation_matix, translated_image.size()); //effettuo la rotazione dell'immagine
	Range rowRange(center.y - windowSize.height / 2, center.y + windowSize.height / 2 + 1),
		colRange(center.x - windowSize.width / 2, center.x + windowSize.width / 2 + 1); //range per il taglio (il +1 alla fine ? perch? l'ultima riga / colonna non viene inclusa nel Range quindi va aggiunta a mano)
	omega(rowRange, colRange).copyTo(omega); //effettua il taglio
	return omega;
}

//media con fourier e derivate (NON USATA)
Mat medianWavelenght(Mat& xSig) { //funzione che usa l'analisi spettrale di fourier (NON COMPLETA)
	Mat medWave(originalImage.rows / dim, originalImage.cols / dim, CV_64F); //matrice delle dimensioni medie calcolate (dimensione (originalImage.rows / 16) x (originalImage.cols/ 16))
	int defaultLenght = 4;
	medWave = defaultLenght;
	
	for (int i = 2; i < xSig.size[0] - 2; i++) {  //la xSig non ? definita per i blocchi che sono distanti a meno di 2 dai limiti esterni
		for (int j = 2; j < xSig.size[1] - 2; j++) {
			Mat v; //vettore della xSig locale
			for (int k = 0; k < xSig.size[2]; k++) {
				v.push_back(xSig.at<double>(i, j, k));

			}
			//cout << "(i, j) = (" << i << ", " << j << ") v =" << v << endl;
			Mat fourier;
			dft(v, fourier);
			
			fourier.at<double>(0) = 0;

			int maxIndex = 1;
			double maxValue = -9999;
			for (int index = 1; index < fourier.size[0]; index++) {
				if (fourier.at<double>(index) > maxValue) {
					maxValue = fourier.at<double>(index);
					maxIndex = index;
				}
			}
			//cout << "index, value = " << maxIndex << ", " << maxValue << endl;

			
			for (int s = 0; s < 32; s++)
				if((s < (maxIndex - 1) || s > (maxIndex + 3)) || (s == 2 * maxIndex))
					fourier.at<double>(s) = 0;
			
			Mat tempXsig;
			idft(fourier, tempXsig);

			/*
			CvPlot::Axes axes = CvPlot::plot(v, "-");
			CvPlot::Window window("v", axes, 480, 640);
			CvPlot::Axes axes2 = CvPlot::plot(fourier, "o-");
			CvPlot::Window window2("fourier", axes2, 480, 640);
			*/ 
			
			Mat der; //matrice della derivata della xSignature locale ricalcolata

			Sobel(tempXsig, der, fourier.depth(), 0, 1);
			vector<int> maxs; //vettore delle posizioni dei massimi trovati
			int sum = 0;
			//cout << "maxs = " << endl;
			for (int p = 2; p < 31; p++) { //il primo e l'ultimo valore della derivata sono sempre 0 quindi li escludo
				bool cond = der.at<double>(p - 1) <= 0 && der.at<double>(p) >= 0 && fourier.at<double>(p) < 0; //condizione per la quale un punto ? un massimo (ATTENZIONE I NERI IN REALTA' SONO VALORI BASSI)
				if (cond) {
					maxs.push_back(p);
					//cout << p << endl;
				}
			}

			for (int p = 1; p < maxs.size(); p++) {
				sum += (maxs[p] - maxs[p - 1]);
			}

			if (maxs.size() == 0)
				medWave.at<double>(i, j) = 1; //se non sono rilevati massimi allora si imposta un valore arbitrario
			else {
				double med = (double)sum / maxs.size();
				if (med < 1)
					med = 1; //questo per evitare errori successivi
				medWave.at<double>(i, j) = med; //larghezza media locale del blocco 16x16 (i, j)
				//cout << "larghezza media locale = " << med << endl;
			}

			//window.waitKey(0);
		}
	} 
	//cout << medWave << endl;
	return medWave;
}

//media con derivate
Mat medianWavelenght2(Mat& xSig) {
	Mat medWave(originalImage.rows / dim, originalImage.cols / dim, CV_64F); //matrice che contiene tutte le larghezze medie delle ridges dei blocchi 16x16
	int defaultLenght = 4;
	medWave = defaultLenght;

	for (int i = 2; i < xSig.size[0] - 2; i++) { //bisogna partire da 2 e finire a (maxCol - 2) perch? la xSignature non ? definita in quei blocchi 
		for (int j = 2; j < xSig.size[1] - 2; j++) {
			Mat v, //matrice della xSignature locale
				der; //matrice della derivata della xSignature locale
			for (int k = 0; k < xSig.size[2]; k++) { 
				v.push_back(xSig.at<double>(i, j, k));
			}

			double xSigMean = mean(v)[0];
			int n = 0;
			Sobel(v, der, v.depth(), 0, 1, 1);
			
			for (int i = 1; i < der.size[0] - 2; i++) {
				if (der.at<double>(i) > 0 && der.at<double>(i + 1) <= 0 && v.at<double>(i) > xSigMean)
					n++;
			}

			if (n == 0) 
				medWave.at<double>(i, j) = defaultLenght;
			else {
				double med = v.size[0] / n;
				if (med < 1)
					med = 1;
				medWave.at<double>(i, j) = med; //larghezza media locale del blocco 16x16 (i, j)
				cout << "larghezza media locale = " << v.size[0] << " / " << n << " = " << med << endl;
			}
			
			/*
			CvPlot::Axes axes = CvPlot::plot(v, "-");
			CvPlot::Window window("v", axes, 480, 640);
			CvPlot::Axes axes2 = CvPlot::plot(der, "-o");
			CvPlot::Window window2("der", axes2, 480, 640);
			window.waitKey(0);
			*/
			
		}
	}
	return medWave;
}

//media con fourier (NON USATA)
Mat medianWavelenght3(Mat& xSig) {
	Mat medWave(originalImage.rows / dim, originalImage.cols / dim, CV_64F); //matrice delle dimensioni medie calcolate (dimensione (originalImage.rows / 16) x (originalImage.cols/ 16))
	int defaultLenght = 4;
	medWave = defaultLenght;

	for (int i = 2; i < xSig.size[0] - 2; i++) {  //la xSig non ? definita per i blocchi che sono distanti a meno di 2 dai limiti esterni
		for (int j = 2; j < xSig.size[1] - 2; j++) {
			Mat v; //vettore della xSig locale
			for (int k = 0; k < xSig.size[2]; k++) {
				v.push_back(xSig.at<double>(i, j, k));

			}
			//cout << "(i, j) = (" << i << ", " << j << ") v =" << v << endl;
			Mat fourier;
			dft(v, fourier);

			fourier.at<double>(0) = 0;

			int maxIndex = 1;
			double maxValue = 0;
			for (int index = 1; index < fourier.size[0]; index++) {
				if (abs(fourier.at<double>(index)) > maxValue) {
					maxValue = abs(fourier.at<double>(index));
					maxIndex = index;
				}
			}
			medWave.at<double>(i, j) = v.size[0] / maxIndex; //calcola la lunghezza media locale delle ridges (dimensione di una semionda della xSig)
			/*
			string windowName = "v";
			CvPlot::Axes axes = CvPlot::plot(v, "-");
			CvPlot::Window window(windowName, axes, 480, 640);

			string windowName2 = "fourier";
			CvPlot::Axes axes2 = CvPlot::plot(fourier, "-");
			CvPlot::Window window2(windowName2, axes2, 480, 640);
			
			cout << "MedWave = " << medWave.at<double>(i, j) << endl;
			window.waitKey(0);
			*/
		}
	}
	return medWave;
}

//funzione che stampa la matrice O delle direzioni delle ridge line in base alla dimensione dell'immagine originale
//@param windowName = nome della finestra che viene mostrata
//@param O = matrice delle direzioni che deve essere stampata a video
void printField(string windowName, Mat& O) {
	int thickness = 1;
	int lineType = LINE_8;
	int length = 5; //lunghezza della linea da disegnare
	Mat output(originalImage.rows, originalImage.cols, CV_8U); //immagine di output
	output = 255;
	auto It = O.begin<double>();
	for (int i = 0; i < O.rows; i++) {
		for (int j = 0; j < O.cols; j++, It++) {
			//cout << "i, j = " << i << ", " << j << endl;
			Point blockCenter(dim / 2 - 1 + dim * j, dim / 2 - 1 + dim * i);
			Point start(blockCenter.x + cos(*It) * length, blockCenter.y - sin(*It) * length),
				end(blockCenter.x - cos(*It) * length, blockCenter.y + sin(*It) * length);
			//cout << "(i, j) = (" << i << ", " << j << ") (start, end) = (" << start << ", " << end << ") *It = "<< *It << endl;
			circle(output, start, 1, (0, 0, 0));
			line(output, start, end, Scalar(0, 0, 0), thickness, lineType);
		}
	}

	namedWindow(windowName, WINDOW_FREERATIO);
	imshow(windowName, output);
	waitKey(1);
}

//funzione che stampa i livelli di grigio della xSignature (per DEBUG)
//@param xSig = matrice della xSignature (dell'intera immagine)
//@param O = matrice delle direzioni
//@param g = immagine normalizzata
void print_xSig(Mat& xSig, Mat& O, Mat& g) {
	
	Mat out(1, 32, CV_64F);
	int length = 5; //lunghezza del vettore che indica la direzione
	for (int i = 0; i < 17; i++) {
		for (int j = 0; j < 11; j++) {
			for (int k = 0; k < xSig.size[2]; k++) {
				//cout << "xSig(" << i << ", " << j << ", " << k <<") = "<< xSig.at<double>(i, j, k) << endl;
				out.at<double>(0, k) = xSig.at<double>(i, j, k);
			}
			string windowName = "X-Signature";
			CvPlot::Axes axes = CvPlot::plot(out, "-");
			CvPlot::Window window(windowName, axes, 480, 640);
			Rect block(j * dim, i * dim, 16, 16);
			Mat originalBlock;
			Mat gBlock;
			originalImage(block).copyTo(originalBlock);
			g(block).copyTo(gBlock);
			Point blockCenter(dim / 2, dim / 2);
			Point start(blockCenter.x + cos(O.at<double>(i, j)) * length, blockCenter.y - sin(O.at<double>(i, j)) * length),
					end(blockCenter.x - cos(O.at<double>(i, j)) * length, blockCenter.y + sin(O.at<double>(i, j)) * length);


			cout <<"(i, j) = ("<<i<<", "<<j<<")    inclinazione (rad) = "<< O.at<double>(i, j) << endl;
			line(originalBlock, start, end, Scalar(0, 0, 0), 1, LINE_8);
			line(gBlock, start, end, Scalar(0, 0, 0), 1, LINE_8);
			circle(originalBlock, start, 1, (0, 0, 0));
			circle(gBlock, start, 1, (0, 0, 0));
			namedWindow("blocco originale", WINDOW_FREERATIO);
			namedWindow("blocco improved", WINDOW_FREERATIO);
			imshow("blocco originale", originalBlock);
			imshow("blocco improved", gBlock);
			window.waitKey(0);
		}
	}
	
	
}