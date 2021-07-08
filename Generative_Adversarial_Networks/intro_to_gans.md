# Learning Goals
* Generative Modeling
* Generative Adversarial Networks
* GAN Variants
    * Deep Convolutional GAN
    * Auxillary Classifier GAN

# Methodology
* Video Lectures
* Source Code
* Coding Along
* Plan for Success

# GANs overview
## Definitions
* Discriminative model: probability that input belongs to a particular class: p(y|x)
    * Usually supervised learning
    * Useful for classification (learn to map an input x to a particular class y)
* Generative model: learn the joint distribution of the input and class labels: p(x,y)
    * Usually unsupervised learning
    * Convert to p(y|x) using probability rules (discriminative model)
    * Useful for creating new data that looks like training data, e.g., images
## Adversarial Training
* Two or more "players" competing against each other
    * Players may be game AIs
* Example: Pacman
* Adversarial search: minimax

* Two models: generator and discriminator
* Discriminator: classify examples as real or fake
* Generator: fool discriminator by generating real-looking examples from random noise
* Analyze situation using game theory (minimax)
* End result: generator creates examples so well that discriminator can't tell real from fake examples
