# Image Recognition with MNIST Notes: Introduction

## Learning Goals
* How does image recognition work?
* What is MNIST?
* How do I build a machine learning model to recognize and classify images with Tensorflow?
* How do I build a machine learning model to recognize and classify images with Keras?

## Prerequisites

* Some Python and numpy knowledge
* A python development environment (course uses Anaconda and Jupyter notebooks)
* An understanding of how machine learning works
* Some experience with Tensorflow
* Ideally Part 1 of this course (Intro to Machine Learning)

## What topics will the course cover?

* Intro to image recognition
* Intro to MNIST
* Build, train, and test an MNIST image recognition model
* Build, train, and test an MNIST image recognition model with Keras

## Why Image Recognition?

* Image recognition is a fun and relatable topic
* There are many practical applications for recognizing and classifying images
* Once we understand how to build the model, it is easy to expand into other areas
* MNSIT is an easy to use dataset

# Part 2: Intro to Image Recognition

## What is image recognition?

* Image recognition is seeing an object or an image of an object and knowing what it is
* Essentially class everything that we see into certain categories based on attributes
* Even if we see something we have never seen before, we can usually place it in some category
* For example, if we see a brand new model of car for the first time, we can tell that it is a car by the weheels, hood, windshield, seats, etc.

## How does this work for us?

* A lot of the time, image recognition for us happens subconsciously 
* We don't necessarily acknowledge everything that is around us
* However, when we need to notice something, we can usually pick it out and define and describe it
* Knowing what something is is based entirely on previous experiences
* Some things we memorize, others we deduce based on shared characteristics that we see in things we do know
* Subconsciously we separate the items we see based on borders defined primarily by differences in color
* Example: It is easy to see a green leaf on a brown tree but hard to see a black cat against a black wall
* We also don't necessarily need to look at every part of an image to know what some part of it is
* Example: if we see only an eye and an ear of someone's face, we know we are looking at a face

# Part three: How does this work for machines?

* Machines do not have infinite knowledge of what everything they see is
* They only have knowledge of the categories we have taught them
* For example, if you create facial recognition, it only classifies images into faces or not faces
* Even the most sophisticated image recognition models cannot recognize everything and have been trained only to look for certain objects
* To a machine, an image is simply an array of bytes
* Each pixel on an image contains information about red, green, and blue color values
* If an image is just black or white, the value for each pixel is simply a darkness value, typcially with 255 as white and 0 as black
* Machines don't care about seeing an image as a whole
* To process an image, they simply look at the values for each of the bytes and look for patterns
* In this way, image recognition models look for groups of similar byte values across images to place an image in a specific category
* For example, high green and brown calues in adjacent bytes may suggest an image contains a tree. If many images all have similar groupings of green and brown values, the model may thing they all contain trees 

# Part four: Tools to help with image Recognition Part 1

## The problem with processing images

* Processing an entire image at a time is a lot
* Most images fed into simple models are small (MNIST images are 28x28 pixels)
* Even this leaves 784 pixels to examine and it's difficult to recognize consistent patterns when comparing all 784 pixel values of one image to another
* Even if we are looking at two images of the same thing, slight position or size cahnges or slightly different shapes could lead to a mislabelling 

## How machines solve this

* Machines solve this problem by first breaking down images into smaller parts and processing them instead
* It starts by applying a sort of a filter to different parts of the image a few pixels at a time to produce seceral smaller, distorted images
* Next, it makes the pieces more abstract by averaging together smaller squares of pixel values
* This ultimately turns this image into something that may not be recognizable by humans but the machines can make sense of it

## Convolutional Neural Networks

* The process of breaking down images and applying the filters is done in a concolutional layer in neural network
* Averaging the values to further distort the images is done through a max pooling layer
* Convolutional Neural Networks get their name from the fact that they have one or more convolutional layers
* These are very popular in image recognition models, although RNNs have also performed well

# Part five: Tools to help with image Recognition Part 2

## Convolutions

* Convulution: an operation on two functions to produce a third function that explains how the shape of one is modified by the other
* Image convultion: applying a kernel or convolution mask to block of pixels to apply an effect
* Ofen used to blur or sharpen images, or detect edges
* A Kernal is a matrix used as a mask to image pixel values
* Kernels are often pre-determined through Tensorflow objects

## Max Pooling

* Max Pooling: replacing a block of pixels by one pixel with the highest value out of the block
* Makes an image more abstract
* Used to downsize an image and also prevent overfitting (drawing conclusions when there are none)
* Generally we specify the size of the matrix and the step size (number of pixels to skip over when applying the next max pool)

## Putting it together

* Almost all convultional neural networks will have a convolutional layer followed by a max pooling layer
* Larger networks will repeat this one or more times along with other layers to perform additional processing
* The result of the two layers is a set of smaller, more abstract images comprised of parts of the original image
* The purpose is to cut out unnecessary image noise and to focus on the stand-out features so that the model can focus on what is important

# Part six: MNIST

## What is MNIST?

* MNIST = Modern National Institute of Standards of Technology
* We will use the dataset that MNIST is famous for
* The dataset contains 70,000 images of handwritten digits
* The even more modern EMNIST dataset was released in 2017 that contains 280,000 images

## How are images Formatted?

* 60,000 trianing images and 10,000 testing images
* Each image is 28x28 pixels (784 total)
* Each image is black and white
* Each image is labelled based on which image it represents
* Each label is in one-hot encoding form
* Instead of a string label, we have an array of 0s and 1s with the 1 in the position that represents the digit and 0s in the rest

## Why bother with MNIST?

* Very highly esteemed data set
* Starting point for many image recognition model
* Great way to learn how to build an image recognition/classification model with a trusted and pre-formatted data set
* Ongoing competition to see who can get the best results



