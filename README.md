# News Headline Classifier

Here we are comparing two classifiers Naive Bayes Classifier or K-Nearest Neighbors Classifier on the same [dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00359/).

A GUI is built in Python3 to Train and Classify data.

The classifier are written from the ground up for text data using Natural Language Processing techniques.

## Prerequisites

* Python
* Pycharm(Not Compulsary but highly recommended)

Install the following packages:

* pandas
* nltk

Installation Instructions: https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html

## Setup Instructions

### Download the Project Files
```
git clone https://github.com/pranavmrane/News_Headline_Classifier.git
cd News_Headline_Classifier
```

### Train and Classify

Run the file titled front_main.py

A interface will open that will allow the user to use the Naive Bayes Classifier or K-Nearest Neighbors Classifier

**Training must be performed for each Classifier atleast one before Classification is Attempted**

For Training, use the dataset present in folder Classifier_Code

Once Training is Completed, Classfication mode can be used any number of times

Use Documentation/sample_input.txt to be used for classification

The documentation provides results. The accuracy for this NB classifier is 82% while KNN classifier has 78% accuracy.

Naive Bayes Classifier performs better and faster for this dataset.
