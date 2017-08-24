#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[signs]: ./examples/exploratary_vizualization.png "Traffic signs example"
[hist]: ./examples/exploratary_label_representation.png "Labels representation"
[new_signs]: ./examples/internet_signs.png "Signs from google"
[topk]: ./examples/topk.png "Top k"
[new_imgs]: ./examples/new_imgs.png "prediction"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

You can explore code in  `Traffic_Sign_Classifier.ipynb`.

###Data Set Summary & Exploration

Dataset consist of images of german traffic signs. All images are in rgb format with size 32x32. Here is example of images in dataset:
![Traffic signs][signs]

Some statistics:
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

Here is the histogram of labels representation:
![Labels representation][hist]

From this picture it's clear that our dataset is quite unbalanced. Calidation and test dataset have same distribution like in training dataset.

To increase number of training examples and add some robustness to small color differences augmentation was applied. Augmentation strategy was to change saturation of image on some random value. After augmentation training dataset size was 4 times bigger than initial. 

###Design and Test a Model Architecture

Simple preprocessing steps for color values consisted in reducing 128 and deviding by 128. I decided not to do grayscaling and not to spend more time on this step and concentrate more on network architecture.
This decision was made because I wanted to preserve color information and hoped on first layer of network to achive better color mapping.

#### Model architecture

I tried different ones, adjust lenet, small residual network with maxpooling after first convolutional layer and last convolutional layer. But the best results I achived with next architecture:

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         						| 32x32x3 RGB image   							| 
| Convolution 1x1         		|  Color mapping. Outputs 32x32x3   							| 
| Convolution 3x3     			| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU							|												|
| Convolution 3x3     			| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU							|												|
| Max pooling 3x3	      			| 3x3 stride,  outputs 10x10x32 				|
| Dropout | 					|
| Convolution 3x3     			| 1x1 stride, valid padding, outputs 8x8x64 	|
| RELU							|												|
| Convolution 3x3     			| 1x1 stride, valid padding, outputs 6x6x64 	|
| RELU							|												|
| Max pooling	3x3      				| 3x3 stride,  outputs 2x2x64 				|
| Dropout | 					|
| Fully connected		|  Output 120        									|
| Fully connected		|  Output 84        									|
| Fully connected		|  Output 43        									|
| Softmax				| Output 43       									|
 

Model was trained on 19 epoches of augmentated training set using adam optimizer on GPU with batch size 128. Dropout probability while training was set to 0.5.

#### Results
My final model results were:
* validation set accuracy of 0.98 
* test set accuracy of 0.973

As a starter I took lenet architecture, but it was hitting something around 91% accuracy on validation. And I need to say it worked quite well, I believe because receptive field was good enough to cover whole 32x32 image. After I played with parameters and added dropout, I was able to get 93% accuracy on validation.

After this I attemped to do deepre resnet like architecture, but got same result. In the end of the day I came to current architecture and got something around 95% accuracy on validation and after adding first convolutional layer which do color mapping I got final results.

###Test a Model on images from internet

Here are five German traffic signs that I found on the web:

![][new_signs]

This is tough examples, because I tried to choose not frequent examples and all this pictures have noise.

None of them was predicted correctly by model. Here is top k accuracy:

![][topk]

And prediction results:
![][new_imgs]

This results are not same as on the test set. I believe way to increase prediction accuracy is to balance data and do better augmentation.
