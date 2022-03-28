# **Generative Adversarial Network using JAX**
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
