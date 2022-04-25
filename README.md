# **Generative Adversarial Network using JAX**
### Andrew Amirov, Thomas Devries
### April 24, 2022
---
### Abstract
 The problem we tackled was from a Kaggle competition titled “I’m Something of a Painter Myself”, which was based around using GANs to transfer the style of Monet’s art across a given set of photos. We decided to implement a cycleGAN to accomplish the task of style transfer. After solidifying our understanding of the architecture we started to implement it using JAX and a supplementary library called Haiku. In the end, we failed to grasp the scope of the project and were not able to get a working JAX implementation.

### Introduction
Our problem was to take photographs and transform them to look like Monet's paintings. This problem falls into the category of style transfer. CycleGANs were created to solve this issue and others like it in 2017. We are going to attempt to implement a CycleGAN using the JAX library. JAX is a deep learning library developed by DeepMind and it's an acronym that stands for Just After eXecution. JAX implements Just-in-Time (JIT) compilation to speed up the code and allow it to be run on the CPU, GPU, and even TPU. JAX does this without needing to make special changes to the code depending on which hardware is being used; unlike in the popular frameworks PyTorch and Tensorflow. Also unlike PyTorch and Tensorflow, JAX is not a framework, it's a library, this gives us a lower level of abstraction, which makes it popular for research purposes.

Our project's value is demonstrating the capabilities of synthetic data generation and spreading awareness of what it can do. Synthetic data can be used to help train models when there is a lack of usable or unique data. Companies like Tesla have been known to use synthetic data while training their self-driving cars. The implementation of this project in JAX could help to quickly create accurate datasets for rare events. Key features of JAX are its ability to be run on any hardware, and its low-level abstraction. While we were able to get some basic convolutions set up for the generator and discriminator in Haiku we underestimated the scope of this project and were not able to complete a full implementation. More details will be provided in the experiments section.

### Related Work
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks([pdf](https://arxiv.org/pdf/1703.10593.pdf))
  - This published work is the basis for CycleGANs. It demonstrates many of the possible use cases for a CycleGAN including Zebras to horses, apples to oranges, and photo’s to paintings. The paper demonstrates many different photo to painting conversions and photo to Monet is one of them.

- CycleGAN implementation from scratch([link](https://www.youtube.com/watch?v=4LktBHGCNfw))
  - This video details how to implement a CycleGAN using the Pytorch library. In the video, Aladdin Persson explains the step-by-step process of creating a CycleGAN that converts images of horses to zebras. We were able to create a proof of concept project based on this and attempted to convert it to a JAX Haiku implementation.

### Data
Our dataset consists of 300 Monet artworks and 7028 photos provided by Kaggle. The images are 256x256 JPEGs. This is what was provided but there was an option to add photos as long as the dataset is less than 10,000 images. We did not do this as our largest hurdle was the train time for a CycleGAN. We did use PyTorch data loaders to transform the data and convert it into an array that was easier to work with.

### Methods
For style transfer problems like ours, the go-to method is a CycleGAN. This problem, converting photos to Monets, actually comes from the paper Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. The paper details the architecture of a CycleGAN and its uses. CycleGANs are able to learn the mapping from a source domain, photo, and convert it into the target domain, Monet. A CycleGAN will also learn the inverse mapping to convert a Monet to a photo. The CycleGAN takes these two mappings, F: P-->M and G: M-->P , and runs (x)-->G(x)-->F(G(x)) with the photo x. This cycle converts a photo, to a Monet, back to a photo. You can then compare the generated photo the original and get a cycle-consistancy loss. Getting the cycle consistency loss for (p)-->G(p)-->F(G(p)) and (m)-->F(m)-->G(F(m)) where p is a photo and m is a monet, we are able to check that the model is preserving the important features in our data. This ensures that we are doing a style transfer and not just generating a random Monte.

 To do a style transfer we need two types of CNNs a Generator that is able to create an image in a certain style, and a Discriminator that is able to do image classification. We then set up generators G and F to map into the domains X and Y as shown below.

 <img src="https://github.com/AndAmir/GAN-with-JAX/blob/main/ReadMEimages/CycleGAN.jpg" height=200>

The discriminators, Dx and Dy, attempt to determine if the generated image fits into the domain.
We then use the generated images to train the discriminators and the discriminators to train the generators. We can also take a deeper look at the architecture of these models.

<img src="https://github.com/AndAmir/GAN-with-JAX/blob/main/ReadMEimages/Generator.jpg" height=600>

The above is the architecture of the Generator model. The first 3 convolution layers downsample the key aspects of the initial image. The first uses 64 filters with 7 by 7 kernels and a stride of 1, the next two layers consist of 128 and 256 filters respectively with each convolution using 3 by 3 kernels and a stride of 2. Then the next 6 layers convert the initial image(Ex. photograph) to the style of the other image(Ex. photograph styled as Monet); each of these convolutions consists of 256 filters with a kernel size of 3 by 3, and a stride of 1. Then the next 2 layers upsample into our final image size using 128 and 64 filters respectively, a 3 by 3 kernel and a stride of 2. The last layer transposes our final image into RGB.

<img src="https://github.com/AndAmir/GAN-with-JAX/blob/main/ReadMEimages/Discriminator.jpg" height=600>

The above is the architecture of the Discriminator. The first four convolutions use 64,128,256, and 512 filters respectively; with 4 by 4 kernels and strides of 2. After each of these layers, there is a leaky ReLU regularization. The last layer consists of 1 filter with a 4 by 4 kernel and a stride of 1. Then a Sigmoid function is used to determine the binary classification.

We looked into alternative approaches but because our training images are not correlated other GANs would not be able to generate the images we need. We did consider using pre-trained models, but for what we needed we could not find any that were compatible with the JAX ecosystem.

### Experiments
For our first experiment, we created a proof of concept in Pytorch to help us understand the requirements for a CycleGAN implementation. We were able to find a helpful guide that used a CycleGAN to solve a different problem and adapt it into a proof of concept. At this point, we noticed a couple of issues with the scope of the project that would make it difficult to complete. On top of the traditional convolutions and activation layers that we learned about in class, this paper required some special functionality in its layers. These included Convolutional Transpose, Leaky ReLU, and Instance Normalization. We were able to figure out how to write a leaky ReLU and understand the functionality of Instance Normalization in getting the final images, but implementing these would be another issue. At this point, we were also unsure how to approach convolutions in JAX.
At this point, we attempted different JAX implementations. We would need some way to do convolutions but the options in JAX didn’t provide layer functionality, just a basic function for a convolution. We attempted to implement from scratch layers but without options for stride, kernels, or any of the important features of a convolutional layer we were not able to set up a working layer. at this point, we consulted our professor who recommended that we use Haiku to work at a high level of abstraction than JAX.

We began implementing our model using Haiku; but when testing, we ran into a problem when using the Sequential method for creating our model. After attempting to troubleshoot we decided to just apply the convolutions manually for each layer by matrix multiplying the result from the previous layer with the current one. With this, we were able to set up convolutional blocks that performed a transform in the forward phase of the algorithm. Haiku also provided us with a Convolutional Transpose and Instance Normalisation. Using this we rewrote the discriminator and generator in Haiku. We tested that it was able to compile without any issues. Unfortunately, we could not figure out how to get the optimization to work with Haiku. Obtaining the weights for optimizing the CNNs seemed like an issue we would not be able to get around. On top of that, we were unsure how the model from PyTorch was approaching the backward phase. It seemed to use a library implementation that we were not able to figure out.

### Conclusion
This project was a serious learning experience in how large frameworks make the very complex tasks seem simple to implement. It was interesting to see how different models used in tandem can solve more complex problems. This experience taught us to be more careful when considering the scope of our project. We chose a project that would be exciting to work on but failed to consider the true scope of our project which ended up being our downfall. In the future, we may be able to revisit this project and implement the features we were not able to get working.

---
# **Proposal**
### Andrew Amirov, Thomas Devries, Bassam Ali
### March 24, 2022

---
We will be tackling the problem of “I’m Something of a Painter Myself”, a Kaggle competition based around using GAN’s to create art. It’s an interesting task because of the dual trainers and the cycling between them.

For context and background, we have to look no further than the Kaggle page and all the resources available within, from recommended tutorials to other groups’ code on the matter; but we’re going to. JAX’s github and reference pages, our destination for everything on JAX and how it functions. We’re also looking at other github pages related to the topic at hand, and for a front to back breakdown of GANs; developers.google has us covered.

https://www.kaggle.com/competitions/gan-getting-started/overview
https://github.com/google/jax
https://jax.readthedocs.io/en/latest/index.html
https://junyanz.github.io/CycleGAN/
https://developers.google.com/machine-learning/gan

For our data, we’re using a dataset of Monet art(300), and a dataset of photos to generate from(7028), both provided by Kaggle, sized to 256x256, in JPEG format. We can add other photos to the generation dataset, so long as the dataset stays under 10,000, and it follows the same formatting rules.

We want to have a style transfer of Monet images to any other image. So we will be looking to implement a CycleGAN. Our reasoning for using CycleGAN as opposed to other GANs is that our dataset consists of uncorrelated images, which CycleGAN is built for these types of problems. We are looking at existing implementations, mainly from the following github repository, https://junyanz.github.io/CycleGAN/ . We will modify this by implementing the algorithm using the JAX framework. Implementing this in JAX will allow us to run the training and inference on either CPU,GPU,or TPU’s; and take advantage of the efficiencies of running it on GPU or TPU’s.

Qualitativly we expect to generate images using our GAN similar to Monet’s paintings. Qualitatively to evaluate our model we will use MiFID (Memorization-informed Fréchet Inception Distance), a modified version of Fréchet Inception Distance (FID). This is a measure of how similar two generated images are with less similar images being better. MiFID expands on FID by accounting for sample memorization and making sure that the images are not too similar to the original data. More information as well as the exact formulas for MiFID can be found [here](https://www.kaggle.com/competitions/gan-getting-started/overview/evaluation).
