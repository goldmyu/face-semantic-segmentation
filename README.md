# Face Semantic Segmentation
A PyTorch implementation to the Face Semantic Segmentation problem, suggested architecture was inspired by the [U-Net Paper](https://arxiv.org/pdf/1505.04597.pdf) [[2]](#2).

The data-set [[1]](#1) used is the FASSEG V2 and is available here - [FASSEG-repository](https://github.com/massimomauro/FASSEG-repository)

## Data pre and post-processing
Pre-processing As our segmentation label images are of shape WxHx3 as they are RGB images, 
and we want to solve a classification of 6 classes problem we need to convert these RGB color of each pixel to a one-hot encoding vector representing the class of that pixel, 
for that we use NumPy location of matching content indices which is extremely efficient (if had more time I would have gone over the entire dataset once, 
convert it to one-hot and stores it to save this time)

Post-processing our output segmentation predictions are of shape 256x256x6, 
in order to visualize them as images we convert them using argmax to get each pixel classified class and build an RGB image from that pixel map that we can display.


## Architecture Selection
As this is a semantic segmentation problem, 
I have decided to design an FCN (Fully Convolutional Network) to solve this problem, 
in an Auto-Encoder like fashion, meaning, the network has two logical parts - a first, 
downsampling part (encoder) and an upsampling part (decoder). 
The problem we are trying to solve is a Per-Pixel classification problem, 
meaning, that for each pixel in the image we want to classify whether it belongs to one of the 6 classes we have - Background Red, Skin Yellow, Hair Brown, Eyes blue, Nose Cayn or Mouth green.
To achieve that we apply a cross-entropy loss to each pixel 6 logits to determine classification and compare to G.T.

The encoder is comprised of conv2d layers, batch_norm, and max pooling with RELU activation function applied after batch_norm after the image has been downsampled into a latent representation, we began to upsample in order to interpolate the data back to the segmentation label image scale. This is done using Conv2dTranspose layers with batch_norm. The network is trained using a batch size of 4 images and a learning rate of 1e-2. It is trained for a maximum of 200 epochs, it took me roughly 30+- minutes to train.  Input shape is 256x256x3 (after images are reshaped) and output is of shape 256x256x6 (reflect num of classes per pixel)

##Results

![Training_epoch_94](results/Training_epoch_94.png?raw=true "Title")

![Training_epoch_98](results/Training_epoch_98.png?raw=true "Title")

![loss](results/train_validation_loss_graphs.png?raw=true "Title")


## References

<a id="1">[1]</a> 
*Khalil Khan*, *Massimo Mauro*, *Riccardo Leonardi*,
**"Multi-class semantic segmentation of faces"**,
IEEE International Conference on Image Processing (ICIP), 2015
-- [**PDF**](https://github.com/massimomauro/FASSEG-repository/blob/master/papers/multiclass_face_segmentation_ICIP2015.pdf)

<a id="1">[2]</a> 
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 
"U-net: Convolutional networks for biomedical image segmentation." 
International Conference on Medical image computing and computer-assisted intervention. 
Springer, Cham, 2015. -- [PDF](https://arxiv.org/pdf/1505.04597.pdf)