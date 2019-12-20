# Business Reviews

## Background
In the text classification tasks, we make use of the textual as well as the non-textual information or meta-data to classify the text which can improve accuracy of statistical model. For example, gender, nationalities, etc.

## Goal
To create a text classification system that classifies user reviews regarding different businesses into "good", "bad", and "average" using a multiple inputs model based on the text of the review along with the associated meta data. 

## Process
* Use ***Keras* Functional API** since it supports multiple inputs and multiple output models.
* Create a deep learning model 
    * Accept multiple inputs
    * Concatenate the two outputs
    * Performing classification using the aggregated input.

## Dependencies
* Pandas
* Seaborn
* Numpy
* Tensorflow
* Keras
* Scikit-Learn


## Dataset
Reviews of different business (yelp_review): https://www.kaggle.com/likarajo/yelp-business-reviews<br>

### Data Analysis and preparation
* The dataset contains a column **Stars** that contains ratings for different businesses. 
    * Have values between 1 and 5. 
* Convert the numerical stars values for the reviews into categorical ones. 
    * Add a new column ***rating*** to our dataset. 
    * Stars = 1 => rating = bad
    * Stars = 2 or 3 => rating = average
    * Stars = 4 or 5 => rating = good





