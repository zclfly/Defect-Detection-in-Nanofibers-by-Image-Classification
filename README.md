# Defect-Detection-in-Nanofibers-by-Image-Classification
 
## Introduction
 
This project concerns the detection of defective regions in SEM (Scanning Electron Microscope) images. These images have been acquired for monitoring the production of nanofibers. The images are contain in the following paper (Carrera2016). Scanning Elector 
Microscope image with anomalies in it. Also, we have the ground truth of the images, calculated also in (Carrera2016).

So far, in (Carrera2016) they have addressed the problem as an anomaly-detection problem, without exploiting during the learning (i.e. training) stage any example of defective regions. So the aim of this project is to address the defect-detection problem as a two-class classification problem where a test image is divided in patches (small squared regions) and each patch is classified as normal/anomalous. In total there are 46 images where 40 of them contains anomalies and 6 are completely normal images. 

So the different aims of the projects are:
- Taking patches based in the GT images where the whole patch is anomalous, or all is normal.
- Training a classifier for predicting between anomalous or normal using a Deep Learning
approach.
- Using this classifier to predict each patch of a new image.

This is a project for the Image Analysis and Computer Vision course at Politecnico di Milano (2016/2017).

## Documentation

In the documentation you could find the explanation of the project. Everything is explained there.

## DataSet

Since this was the continuation for a paper, I don't know if I'm able to public the dataset I've created. If possible, it would be uploaded in the future.

## Author

Francisco Carrillo Pérez (C)

< carrilloperezfrancisco@gmail.com >

