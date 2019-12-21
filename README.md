# Business Reviews
Deep Learning model to predict ratings of business reviews

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
* Matplotlib

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

## Data Preprocessing
* Remove punctuations and numbers
* Remove single characters
* Replace multiple spaces with single space

# Text-only Model
* Convert text label to one-hot encoded vector
    * covert text to integer labels
    * convert the labels to categories
* Split data into training and test sets
* Create Word embeddings
    * convert text into token sequences
    * pad each sequence to equal length
    * load pre-trained GloVe file with word vectors from the [Stanford Core NLP glove file](https://nlp.stanford.edu/projects/glove/)
    * create embedding matrix using the word vectors from the GloVe file
* Create the model using ***Keras* functional API**
    * One Embedding (input) layer
    * One LSTM layer
        * 128 neurons
    * One Dense (output) layer
        * 3 neurons (for 3 outputs bad, average, good)
        * Activation function = Softmax
    * Compilation
        * Loss function = categorical cross-entropy
        * Optimization fucntion = Adam
        * Metric = Accuracy
* Train the model
    * Train data = 80%
    * Validation data = 20%
    * Epochs = 10
* Evaluate the model
    * Test score
    * Test accuracy

---

# Metadata-only Model
* Convert text label to one-hot encoded vector
    * covert text to integer labels
    * convert the labels to categories
* Split data into training and test sets
* Create the model using ***Keras* functional API**
    * input layer
        * 3 metadata
    * two hidden dense layers
        * neurons = 10
        * activation = Relu
    * output dense layer
        * neurons = 3 for 3 outputs (bad, average, good) 
        * activation = Softmax
    * Compilation
        * Loss function = categorical cross-entropy
        * Optimization fucntion = Adam
        * Metric = Accuracy
* Train the model
    * Train data = 80%
    * Validation data = 20%
    * Epochs = 10
* Evaluate the model
    * Test score
    * Test accuracy
<br>
NOTE: The accuracy can be increased by increasing the number of dense layers or by increasing the number of epochs.

---

# Model with multiple inputs
* Input is the **combination of both textual information and meta information**
* Built with ***Keras* Functional API**
* Create two sub-models

## Sub-model 1
* Input = textual: review text
* Embedding layer
* LSTM layer: 128 neurons
* Output = Dense: 3 neurons (bad, average, good)

## Sub-model 2
* Input = meta info: useful, funny, cool
* Dense layer 1: 10 neurons
* Dense layer 2: 10 neurons
* Output = Dense: 3 neurons (bad, average, good)

## Concatenate sub-model outputs
* Input = Sub-model 1 output + Sub-model 2 output
* Dense layer: 10 neurons
* Output = Dense: 3 neurons (bad, average, good)

---

# Conclusion and Improvements
* If the differences for loss and accuracy values is minimal between the training and test sets, then the model is not overfitting.
* We only used 10,000 rveiew records due to hardware constraint. Better performance can be achieved by **training the model on a higher number of records**.
* **More LSTM and dense layers can be added to the model**. If the model overfits, then layers can be dropped out as necessary.
* **The optimizer function can be changed**.
* The model can be **trained with higher number of epochs**.
