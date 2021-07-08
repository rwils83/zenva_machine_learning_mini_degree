# Applied Deep Learning
## Topics
* Input data normalization & dataset augmentation
* Regularization (Weight Sparsity, Weight Decay, Dropout)
* Overfitting
* Early Stopping
* Cross Validation
* Parameter Update Algorithms
* Spatial Pyramid Pooling
* Batch Normalization
* Weight Initialization
* ... More than could fit on the slide? 

## Dataset Augmentation
* For large networks, we need lots of training data but don't have it
* Collecting training data is expensive, so synthesize more from existing training set 
* Operations: 
    * Rotations
        * Just rotating an image 3 degrees creates a "new image" according to Convolutional networks
    * Scaling and cropping
    * Flipping/reflections
    * Noise
    
## Data Preprocessing and cleaning
* Zero mean and unit variance(normalization) across batches
* Principal Components Analysis (PCA)
    * Reduce dimensionality of input
    * Especially for unsupervised data
* Whitening/Sphering
    * Decorrelate and unit variance
    * Run after PCA

## Monitoring Learning
Grab the slides

