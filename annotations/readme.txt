The annotation files (anno_a.mat, anno_b.mat) contain the bounding box 
coordinates (upper left and bottom right point) that have been used to crop 
the images of the PRID 2011 dataset out of the videos. The cropped images 
have been resized to a common size of 128x64 pixels. The order of persons 
in the annotation files is the same as in the dataset, i.e., the first 200 
persons appear in both camera views.

Both files contain a list over persons stored in the variable "anno". Each 
list entry consists of lists holding the frame indices where the person 
appears in the video (the first video frame has index 1) and the upper left 
(ulx, uly) and bottom right points (brx, bry) of the corresponding bounding 
boxes.

Example for cropping the 1st appearance of person 10:

frame_idx = anno(10).frame(1);
bb = [anno(10).ulx(1), anno(10).uly(1), anno(10).brx(1), anno(10).bry(1)];


Contact:

Martin Hirzer: hirzer@icg.tugraz.at
Csaba Beleznai: csaba.beleznai@ait.ac.at
Peter M. Roth: pmroth@icg.tugraz.at


Please cite the following paper if you use this dataset:

Person Re-Identification by Descriptive and Discriminative Classification
Martin Hirzer, Csaba Beleznai, Peter M. Roth, and Horst Bischof
In Proc. Scandinavian Conference on Image Analysis, 2011
