# Segmentation with Linknet
## Dataset
### [Cityscapes Pair](https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs)
This dataset has 2975 training images files and 500 validation image files. Each image file is 256x512 pixels, and each file is a composite with the original photo on the left half of the image, alongside the labeled image (output of semantic segmentation) on the right half.
![example](https://github.com/Trong1102/Segmemtation-With-Linknet/assets/86673103/6a78167a-a78d-4d0b-9bbd-b2ca482d45f0)

### [Brain Tumor](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)
This dataset has 6128 images files include 2 folders images and masks. Each image file is 512x512 pixels, and the order of pictures in the file 'images' corresponds to the labels in the file 'masks'

![1](https://github.com/Trong1102/Segmemtation-With-Linknet/assets/86673103/8a241bc6-e8ec-480e-82cf-fb7861d3fbb0)

![1](https://github.com/Trong1102/Segmemtation-With-Linknet/assets/86673103/7bff337d-11ce-4ed8-a931-5f9c455524bc)

## Model
![image](https://github.com/Trong1102/Segmemtation-With-Linknet/assets/86673103/5a9c6f7c-d925-4e1b-9720-39dd2b97b5fd)

This is architecture of Linknet include 4 encode blocks corresponding to 4 decode blocks and 3 shortcut connections.

Encoding part:
- The LinkNet model uses a convolutional network such as ResNet or similar architectures to extract features from the input image.
- Convolutional layers and downsampling are used to reduce the size of the feature and extract more abstract information when going deeper into the network.

Shortcut Connections:
- The LinkNet model uses short-cut connections to pass information from the encryption layers down to the decryption layers.
- The short path helps to pass detailed information from the encoding layers down to the decoding layers, retaining important local information and minimizing information loss during segmentation.
- Information from encryption layers is combined with information during decryption through short-path interconnection layers, resulting in more detailed and accurate segmentation results.

Decoding part:
- The decoding part in the LinkNet model uses upsample layers to reconstruct the image size and produce detailed segmented output.
- Upsample layers help to increase the size of the feature to below the size of the original image.
- Information from the encoding layers is transmitted over a short path and connected to the decoding layers to combine information from high- and low-level features, producing the final segmentation result.

## Train model use Colab
### [Train model with Cityscapes dataset](https://github.com/Trong1102/Segmemtation-With-Linknet/blob/main/Brain_tumor.ipynb)
### [Train model with Brain tumor dataset](https://github.com/Trong1102/Segmemtation-With-Linknet/blob/main/Cityscapes.ipynb)
