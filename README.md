# Image-Recognition-Handwritten-Alphabets-
This project demonstrates of building a image recogniser using self coded Inception Model and demonstrates various techniques of data augmentation in case of less training data.  
# Process
I have divided this project in to 4 parts
## 1) Data loading and preparation
## 2) Data augmentation
## 3) Coding for CNN(InceptionNet)
## 4) Model Training

# DATA LOADING AND PREPARATION
data is taken from kaggle and the obtainied in CSV file which had shape of (372450,785),out of which column 0 (in python indexing starts from 0)
is the target variable.\
when the distribution of the data was examined I found that it is highly imbalannced dataset.\
Some of the alphabets out of 26 alphabets had very less data so I decided to go for data augmentation.\
As we have 784(leaving target variable) features so we can convert it to a (28X28X1) image and feed it to the CNN.

# DATA AUGMENTATION
well this is the most important part as of now because we've less data in for most of the labels so it is better to create 
artificial images using image augmentation methods for those labels.\
The methods used are\
**1)zooming**\
**2)standardization**\
**3)random_rotation**\
**4)Adding Gaussian Noise**\
**5)Shifting**\
**Didn't used flipping and croping because the filpped letters will not be the same and same applies for cropping.**\
**6) Used PCA for dimensionality reduction.(could have used AUTO ENCODER,but it is computationaly expensive.)**

# Modelling
**we used hand coded Inception Model using just two layers.**\
## Inception Model Description
**1X1 filters of 96 in numbers and it is convolved with the dataset and further it is convolved with 3X3 filter of 96 in number.**\
**The Second parallel layer contained 1X1 filters of 32 in numbers and it is convolved with the dataset and further it is convolved with 5X5 filter of 32 in number.**\
**The third parallel layer was of only 96 1X1 filters  and it is convolved with the dataset.**\
**The last parellel layer contained max_pooling with same convolution resulting in preserving the dimension of the images and the pooled features are further convolved using 1X1 filters of 32 in number.**\

**All these outputs features are concatenated on depth and the resulting dimension was 28,28,256. From no channel we reached to 256 channels with the use of ionception layer.**\
I used the same architecture for the 2nd inception layer and resulted dimension was 28X28X1280.\
I further use some convolutional layer with strides of 2 and pool of 2 and 3 respectively.\
Then a fully connected layer with 26 outputs.\
I used drop_out of 0.7 and batch_normalization in each layer.

# Training
**Choice of Activation function:-- ReLU for hidden layers and softmax for output layer**\
**Choice of Optimization function:---Adam**\
**Choice of Loss function:--- softmax_with_cross_entropy_with_logits_v2 ad it was a multiclass classification problem.**\
**Hyperparameters are described in notebook**\
**Accuracy metric was reduce_mean(squared_error) of tensorflow**\
**Achieved an accuracy of 77%(just 10 epochs) on the validation set.**\
Note:---\
This model was trained on **GOOGLE COLAB** and due to lots of parameters it was taking huge time so data is divided in 30 batches,and trained for only 10 epochs.\

**Aim of this project was to understand the working of **INCEPTION MODEL** which was self coded as a purpose of experiment we would have use transfer learning but it was rather more exciting.**
## Challanges:--
1) Slow training (Should go for DASK or some big data frame works or training in batches)\
2) Less data for some classes (data Augmentation is expensive process when applied to large datasets)


## Further improvements
1) Advanced data augmentation using GANs and Neural Style Transfer.

**Thanks**
