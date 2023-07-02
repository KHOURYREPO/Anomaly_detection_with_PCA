# Anomaly_detection_with_PCA


This repo is to develop a fault detection system for a sensors network measuring air pollutants and to localize on line the faulty sensor. 

## Database

Load the file named training.csv. It contains two matrices X1 and X2.
Each matrix contains pollutant concentrations (ozone, nitrogen oxide, and carbon dioxide) measured in 6 locations around the town of Nancy. Measurements are collected every 15 minutes. The matrix is formed by 18 rows, the first 3 rows correspond to measurements of ozone, nitrogen oxide, and carbon dioxide in the first location, the 3 following rows to the measurements in the second location, and so on.
X(i,:) represents the measurement of the ith sensor in the function of time.
X(:,i) represents the set of measurements from the 18 sensors at a given time.
X1 corresponds to a recording when no fault occurred on the system. X2 corresponds to a recording when a fault occurred on a sensor.


