# Pix2PixNIfTI

Extension of [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to run with NIfTI files (2D or 3D).
Pix2Pix is a general framework for image-to-image translation.
With our extension this cGAN can learn a mapping between two MRI sequences.

The general framework is still the same, so the that can be found in [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) is still valid.
However, the code has been extended with the following:
* We modify the loading functionality to accept 3D NIfTI images.
* We extend the data augmentation process with an additional library, [TorchIO](https://github.com/fepegar/torchio), which implements useful preprocessing and data augmentation routines for medical imaging.
* We add the mixed precision training functionality from [NVIDIA APEX](https://github.com/NVIDIA/apex), which can be enabled with `--fp16`.
* Since a big portion of our scans is background, the L1 loss is only computed on the actual brain voxels.
* Linear additive upsampling can be used instead of the transpose convolution operation. Select the desidered upsampling with `--upsampling [deconvolution | linear]`.
* We modify the loss function to consider, additionally to the L1 loss, the MSE loss of the pixels corresponding to the tumor ground truth (if available).

## Installation
The framework is written in python3 and requires some libraries, which can be installed with
```
pip install requirements.txt
```

## Running the program
The model can be trained with `train.py` and then the desired data set can be tested with `test.py`.

### Data and Folder Organization
This software accepts all the formats previously accepted by Pix2Pix and additionally accepts 3D NIfTI files. 
The MRI scans should be skull-stripped and registered to the same template.

For the NIfTI case we require that there is a separate folder for the source and the target sequence.
The folder structure should have a `path/to/data` folder where there is a `train` folder that contains the training data
and some test/validation folders. Each of these folders has one folder for each sequencing, so `t1`, `t2`, `t1ce`, `flair` 
and one for the ground truth `truth`.
If you are only interested in the translation between `t1` and `t2`, the other folders are not needed.
The tumor ground truth `truth` is used to compute the MSE of the tumor, and will be ignored if not present.
```
path/to/data
|
└───train
│   └───t1
│   |   │   pat1.nii
│   |   │   pat2.nii
│   |   │   ...
|   |   |   patn.nii
|   |
│   └───t2
│   |   │   pat1.nii
│   |   │   pat2.nii
│   |   │   ...
|   |   |   patn.nii
|   |
│   └─── ...
└───val
│   └───t1
│   |   │   pat1.nii
│   |   │   pat2.nii
│   |   │   ...
|   |   |   patn.nii
|   |
│   └─── ...
```

Normally, the data sets containing MRI scans have a folder `data/to/fix` 
which contains folders like `train` and `test`. These folders contain one patient folder each, and each patient folder
contains `t1`, `t2`, `t1ce`, `flair` and `truth`. 
Given a file hierarchy of this type, we created a script that takes the 
images and saves them such that they can be used by Pix2Pix.
```
python ./datasets/fix_data.py --folder data/to/fix --save_folder path/to/data --phase train
```
The `--phase train` argument can be changed to any folder name.

### Train

To start the actual training procedure run
```
python train.py --dataroot path/to/data --dataset_mode nifti --model pix2pix3d --name test_name
```
Which will create a model for 3D MRI scans. To use 2D slices instead select `--model pix2pix`.
To select a slice to run the training on, use the parameter `--chosen_slice`. 
At the moment only transverse slices are supported and the default slice is the middle transverse slice for a scan of size 240x240x155.

The network is a `unet_128` by default, so the images should have side lengths which are multiples of 128. 
However, if this is not the case, random crops will be performed to ensure that the entire structure of the brain is captured.
In our tests `resnet_6blocks` performed best, and we think they are the best choice for this purpose.

### Test
Once the model has been trained, we can retrieve the results for the `test_name` experiment with
```
python test.py --dataroot path/to/data --dataset_mode nifti --model pix2pix3d --name test_name --preprocess resize --excel --postprocess 1 --phase name/to/test
```
If the source images do not have side lengths which are multiples of 128 they can be padded with `--preprocess pad`.

The flag `--excel` prints the results of the MSE computation to a csv file. 
Otherwise, the results are only printed to screen.

`name/to/test` is the name of the folder where the MRI images for testing are located. For example, if the files to test are in `/path/to/data/name/to/test`, then `name/to/test` should be used. 
The default value is `test`.

The variable `--postprocess` can be used to postprocess the image:
* **-1** (default): means no post-processing,
* **0**: scales the image to the [0, 1] range,
* **1**: standardizes the image to have zero mean and unit variance,

It is also possible to select a filter to apply to the resulting image with `--smoothing`. 
Both the original and the filtered image will be saved as output.
The options are `median` and `average`, where `median` applies the median filter while `average` applies a simple averaging filter.
The default filter is `median`.

In case you not have a GPU, please add the option `--gpu_id -1`.

### Docker

The file `base.dockerfile` contains the base structure of our Pix2PixNIfTI image and it can be used both for training and testing.
Some example dockerfiles that build on top of that image can be found in `scripts`.

Here an example of how to create the base image and some images for training and testing.
```
# Fix training dataset
python3 ./datasets/make_nifti_dataset.py --folder /path/to/data --save_folder /fixed/pix_data --phase train
# Fix validation dataset
python3 ./datasets/make_nifti_dataset.py --folder /path/to/data --save_folder /fixed/pix_data --phase validation

# Build Pix2Pix base image
docker build --force-rm -f base.dockerfile -t pix_base .

# Run Training ResNet
docker build --force-rm -f ./scripts/train_resnet.dockerfile -t train_resnet . && \
nvidia-docker run --rm --mount type=bind,source=/fixed/pix_data,target=/input \
--mount type=bind,source=/location/pix_checkpoints,target=/checkpoints \
--mount type=bind,source=/location/pix_output,target=/results train_resnet

echo "Finished ResNet Training"

docker build --force-rm -f ./scripts/val_resnet.dockerfile -t val_resnet . && \
nvidia-docker run --rm --mount type=bind,source=/fixed/pix_data,target=/input \
--mount type=bind,source=/location/pix_checkpoints,target=/checkpoints \
--mount type=bind,source=/location/pix_output,target=/results val_resnet

echo "Finished ResNet Validation"

python3 ./datasets/make_nifti_dataset.py --folder path/to/data --save_folder /fixed/pix_data --phase testBraTS
python3 ./datasets/make_nifti_dataset.py --folder path/to/data --save_folder /fixed/pix_data --phase testUK
python3 ./datasets/make_nifti_dataset.py --folder path/to/data --save_folder /fixed/pix_data --phase testNoTruth

docker build --force-rm -f ./scripts/test_resnet.dockerfile -t test_resnet . && \
nvidia-docker run --rm --mount type=bind,source=/fixed/pix_data,target=/input \
--mount type=bind,source=/location/pix_checkpoints,target=/checkpoints \
--mount type=bind,source=/location/pix_output,target=/results test_resnet

echo "Finished ResNet Testing"
``` 

## Citation

If you use this repository, please cite this article:

```
Baldini, G., Schmidt, M., Zäske, C., Caldeira, L.L. (2024). MRI Scan Synthesis Methods Based on Clustering and Pix2Pix. In: Artificial Intelligence in Medicine. AIME 2024. Lecture Notes in Computer Science, vol 14845. Springer, Cham. https://doi.org/10.1007/978-3-031-66535-6_13
```

The article is also available on [ArXiv](https://arxiv.org/abs/2312.05176).
