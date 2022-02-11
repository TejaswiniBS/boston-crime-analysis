# Exploratory Data Analysis, Crime Type and Crime Severity Prediction for Boston Crime Dataset

## I. Abstract
In recent years there is a substantial increase in the number of crimes that occur on a daily basis across the globe. With this increase, the amount of crime data is also increasing. This makes it crucial to come up with efficient techniques to predict the occurrence of crimes. This project conducts and EDA on the dataset to analyze the trends of crime, locations of high criminal activity, and hotspot of crimes over the years. Next, several machine learning models like K Nearest Neighbor, Decision Tree Classifier, Random Forest Classifier, Extra Tree Classifier and a benchmark classification model are utilized to generate accurate predictions of the crime type as well as severity of crimes using spatial and temporal data. Finally, techniques like oversampling, under sampling of data for KNN classifiers, and hyper parameter tuning will be employed to further improve the accuracy of crime type prediction. The model performance is evaluated using a Multi Class Classification Evaluator.

## II. Dataset
https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system

## III. Introduction
Inspired by the concept of Predictive Policing that was employed in Santa Cruz, California, that brought down the number of burglaries by 19%, this project deals with analyzing the incidents reported by the Boston Police Department. Boston is one of the largest and most populated cities in the state of Massachusetts. As of 2017 it has a population of 685,094 and has almost 19 million visitors each year.According to the FBI Uniform Crime Reporting Statistics, Boston sees 655 violent crimes per 100,000 making it less safer than 83% of cities in the United States. This makes it crucial to come up with efficient techniques to predict the occurrence of crimes so that the related officials can take up precautionary measures to prevent them from happening. Additionally, this information could aid hospitals, community as well as tourists to the city. This project consists of the following analyses:
- ##### Exploratory Data Analysis
  1. Dangerous districts in Boston.
  2. Crime rates during different seasons.
  3. Crime rate over the years
  4. Crime rates during different hours of the day
  5. Timelapse of crime hotspots in Boston
- ##### The crime type based on temporal, spatial, and a combination of spatial and temporal data
  1. Dataset without combining the crime types
  2. Dataset where similar crime types were combined
- ##### The severity of the crime that could occur based on the temporal, spatial, and a combination of both spatial and temporal data.
  1. Severity prediction based of crime type
  2. Severity prediction based on UCR_PART

## IV. Tools and Technology Used
- PySpark
- WEKA
- Numpy
- Pandas
- Sklearn
- Seaborn, Matplotlib, Folium

## V. Data Preprocessing
  1. Data Cleaning: Removed duplicate records, and records with null values for lat and long
  2. Discretization: Removed non criminal offense types, and combined similar crimes to for a new category.
  3. Handling Outliers: Data objects corresponding to crimes that occurred outside Boston were removed.
  4. Feature Creation: Depending on the type of analysis being conducted, multiple features like day/night, seasons etc. were created using the existing features.
  5. Feature Selection: retained all the relevant features or attributes and dropping those that would be irrelevant for further analysis
