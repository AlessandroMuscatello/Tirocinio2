# Fingerprint minutiae extraction via grey scale image
Questo progetto si basa sul lavoro compiuto da Devansh Arpit e Anoop Namboodiri nella loro pubblicazione "*Fingerprint feature extraction from gray scale images by ridge tracing*" e ha lo scopo di creare un programma che permetta l'estrazione di minuzie da un'impronta digitale analizzando direttamente l’immagine in scala di grigi ed effettuare un tracciamento delle creste dell’impronta.

# Librerie necessarie
* OpenCV (v 4.5.2) per la manipolazione delle immagini
* CvPlot (v 1.2.1) per grafici di debug

# Installazione
L'ambiente di sviluppo utilizzato è Microsoft Visual Studio.
Per l'installazione delle librerie si farà riferimento alle guide ufficiali:
### OpenCV (v 4.5.2)
1. Scaricare la libreria dal sito opencv.org e installare la libreria nella cartella C:\(OPENCV FOLDER)\
2. Aggiungere ai path di sistema C:\(OPENCV FOLDER)\build\x64\vc15\bin
3. Una volta creato il progetto in Visual Studio aprire le proprietà del progetto.
(IMG1/2)
4. Andare in Configure Properties/C-C++/Additional Include Directories, premere sulla freccia a destra poi <Edit...>.
(IMG3)
5. Cliccare su "New Line" e aggiungere la riga "C:\(OPENCV FOLDER)\build\include". Cliccare OK.
(IMG4)
6. Andare in Configure Properties/Linker/General
(IMG5)
7. In Additional Dipendencies aggiungere la riga "opencv_world452d.lib". Cliccare OK.
(IMG6)
8. In Additional Library Directory aggiungere la riga "C:\(OPENCV FOLDER)\build\x64\vc15\lib"

### CVPlot (v. 1.2.1)
La libreria è già inclusa all'interno del progetto.
Seguendo i passaggi dell'installazione della libreria OpenCV

1. Aprire le proprietà del progetto in Visual Studio.
2. Andare in Configure Properties/C-C++/Additional Include Directories
3. Aggiungere "(PROJECT FOLDER)\cv-plot-1.2.1\CvPlot\inc"
4. Andare in Configure Properties/Linker/General/Additional Library Directories
5. Aggiungere "(PROJECT FOLDER)\cv-plot-1.2.1\CvPlot\inc"
