# Creative Networks
*Various models attempted at generating "creative" outputs*

Current models:
- ArtGAN
Future options:
- Cycle-GAN
- Art-style Autoencoder
- Text generation network

## ArtGAN
*A DCGAN trained on art images to produce new outputs*

So far, this has been a really tricky project: the current DCGAN is learning at a stable rate, and creating relatively organic patterns / forms, but as of yet, nothing that would be intuitively classified as art.

The largest difficulty (as is often the case with GANs) has been introducing sufficient complexity into the network whilst maintaining sufficient balance between the *generator* and *discriminator*.


### ArtGAN v1
An initial fist pass (based on a GAN previously developed on the MNIST dataset, and then scaled up to introduce further layers for greater resolution) made some really interesting images in terms of shapes / colours, however quickly collapsed to a point where the discriminator was failing to make any progress. Ultimately, any tweaks let to a complete divergence on the networks, with either the *generator* or *discriminator* network falling into an extremely high loss function and no longer learning.

![v1 ArtGAN image 1](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-1.jpg)
![v1 ArtGAN image 2](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-2.jpg)
![v1 ArtGAN image 3](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-3.jpg)
![v1 ArtGAN image 4](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-4.jpg)
![v1 ArtGAN image 5](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-5.jpg)
![v1 ArtGAN image 6](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-6.jpg)
![v1 ArtGAN image 7](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-7.jpg)
![v1 ArtGAN image 8](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v1-8.jpg)

It's unclear whether these images where authentic learnings from the abstract-art training set, or whether they were simply interesting patterns generated by a complex network. Based on the very early stage (within 20-30 minutes training) at which these patterns appeared, it's highly likely that it's the second.


### ArtGAN v2
A second version of this network had more success- the network is based on the keras GAN documentation for reproducing celebrity faces without changes made to the network itself, but some tweaks to the api to allow easier saving and loading of weights so that the network can be easier trained in *sessions* (i.e. on a local machine without an 'always on server').

Training of this network has been stable on an abstract art dataset without leading to divergence of the two networks. However, images *seem* to have stabilised around the point of the organic patterns shown below. It may be the case that continuing to train the network will ultimately lead to an improvement in the output, as the two networks have not yet reached a point of final convergence.

However, within epochs/training batches, all produced images are of the same exact colour-palette, potentially an indication of mode collapse. It may now be the case that the *generator* has learned to produce organic patterns, and is now cycling through colour palettes to avoid the *discriminator* detecting repeating colors.

![v2 ArtGAN image 1](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-1.png)
![v2 ArtGAN image 2](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-2.png)
![v2 ArtGAN image 3](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-3.png)
![v2 ArtGAN image 4](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-4.png)
![v2 ArtGAN image 5](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-5.png)
![v2 ArtGAN image 6](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-6.png)
![v2 ArtGAN image 7](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-7.png)
![v2 ArtGAN image 8](https://github.com/benrutter/creative-networks/blob/main/readme_images/artgan-v2-8.png)


### ArtGAN v3
I'm now experimenting with ways of improving the complexity and depth of the model. So far, making some even relatively simple changes has lead to instability of the GAN network. For example:

- Adding batch normalisation
- Changing kernal size to odd numbers
- Introducing dropout

Led to the *discriminator* outperforming the *generator* to the point where the *generator* stopped learning and collapsed into producing completely yellow squares.

One potentially promising option is switching to a Wasserstein loss function, as this may allow both networks (importantly the GAN) to learn without requirign complete balance between the two.
