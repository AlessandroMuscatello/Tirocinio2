#include <opencv2/opencv.hpp>
#include "improveImage.h"
#include "newPoint.h"
#include "tracing.h"
#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

//matrici globali invarianti
Mat originalImage; //matrice dell'immagine originale
Mat filteredImage; //immagine originale filtrata dall'algoritmo CLAHE
Mat medianLenght; //matrice che contiene le dimensioni medie delle ridges calcolate in blocchi 16x16 pixel
Mat O; //matrice delle orientazioni (oriented field)


int main() {

	
	//lettura dell'immagine
	originalImage = imread("miura.bmp", IMREAD_GRAYSCALE);	//CV_8UC1

	//controllo di avvenuta lettura
	if (originalImage.empty())
	{
		cout << "Could not open or find the image." << endl;
		system("pause"); //attende la pressione di un qualunque tasto per proseguire
		return -1;
	}
	//taglio l'immagine originale in modo che sia divisibile in blocchi da 16x16 pixel
	originalImage = originalImage(Range(0, originalImage.rows - originalImage.rows % 16), Range(0, originalImage.cols - originalImage.cols % 16));
	
	medianLenght = getMedianWaveLenght(); //matrice che contiene le dimensioni medie delle ridges calcolate in blocchi 16x16 pixel
	filteredImage = claheFilter();
	O = getOrientedField();
	Mat finalImage(originalImage.rows, originalImage.cols, CV_64F); //immagine finale
	Mat finalPath(originalImage.rows, originalImage.cols, CV_64F); //estrazione dei path
	finalPath = 255;

	imshow("original", originalImage);
	namedWindow("filtered", WINDOW_FREERATIO);
	imshow("filtered", filteredImage);
	namedWindow("finalImage", WINDOW_FREERATIO);
	resizeWindow("finalImage", Size(2 * originalImage.cols, 2 * originalImage.rows));
	namedWindow("finalPath", WINDOW_FREERATIO);
	resizeWindow("finalPath", Size(2 * originalImage.cols, 2 * originalImage.rows));
	printField("O", O);
	waitKey(1);


	Point p;
	for (int i = 20; i < originalImage.rows - 20; i += 15) { //ricerca dei seeding points
		for (int j = 20; j < originalImage.cols - 40; j += 15) {
			p = Point(j, i);
			cout << "p = " << p << endl;
			if (isSeedingPoint(p, finalImage)) { //controllo che p sia un possibile seed point
				traceLine(p, finalImage, finalPath); //modifica l'immagine finale aggiungendo i punti alla linea con seed p
			}
		}
	}
	waitKey(0);
	
}