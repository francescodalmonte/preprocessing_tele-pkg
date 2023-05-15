dir = getDirectory("Choose a Directory ");

datpath = substring(dir,0,lengthOf(dir)-5) + ".dat"
open(datpath);



path = dir + "/VistaAcq001_003.bmp"
open(path);

path = dir + "/VistaAcq001_004.bmp"
open(path);


imageCalculator("Subtract create 32-bit", "VistaAcq001_003.bmp","VistaAcq001_004.bmp");
selectWindow("VistaAcq001_004.bmp");
close();
selectWindow("VistaAcq001_003.bmp");
close();
