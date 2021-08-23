#include <opencv2/opencv.hpp>
#include "improveImage.h"
#include "newPoint.h"
using namespace cv;
using namespace std;

Mat originalImage;


int main() {

	
	//lettura dell'immagine
	originalImage = imread("m 3.bmp", IMREAD_GRAYSCALE);	//CV_8UC1

	//controllo di avvenuta lettura
	if (originalImage.empty())
	{
		cout << "Could not open or find the image." << endl;
		system("pause"); //attende la pressione di un qualunque tasto per proseguire
		return -1;
	}
	//taglio l'immagine originale in modo che sia divisibile in blocchi da 16x16 pixel
	originalImage = originalImage(Range(0, originalImage.rows - originalImage.rows % 16), Range(0, originalImage.cols - originalImage.cols % 16));
	
	Mat medianLenght = getMedianWaveLenght(); //matrice che contiene le dimensioni medie delle ridges calcolate in blocchi 16x16 pixel
	Mat filteredImage = claheFilter();
	Mat O = getOrientedField();
	Mat finalImage(originalImage.rows, originalImage.cols, CV_64F);
	Point p;

	for (int i = 10; i < originalImage.rows - 10; i += 10) {
		for (int j = 10; j < originalImage.cols - 10; j += 10) {
			p = Point(j, i);
			//cout << "p = " << p << endl;
			if (isSeedingPoint(p, filteredImage, finalImage, medianLenght, O)) {
				
			}
		}
	}
	
}