# YOLOv8 Training & Inference Scripts for Bounding Box and Segmentation

This repository is your guide to training detection models and utilizing them for generating detection outputs (both image and text) for bounding box detection and pixel segmentation tasks.

> [!IMPORTANT]
> Requires **Python 3.11** or newer.

## Installation Instructions for Windows
1. **Clone the Repository**
   
Clone this repository to your local machine using Git:
`git clone https://github.com/MNeMoNiCuZ/yolov8-scripts`


2. **Navigate to the Repository Folder**
   
Change into the cloned repository's directory:
`cd yolov8-scripts`

3. **Clone Ultralytics Repository**
   
Clone the Ultralytics repository within the `yolov8-scripts` folder:
`git clone https://github.com/ultralytics/ultralytics`


4. **Run the Setup Script**
   
Execute `setup.bat` by double-clicking on it. When prompted, enter an environment name or press Enter to accept the default. This script creates a virtual environment inside the Ultralytics folder, sets up necessary directories, and activates the environment.

5. **Install PyTorch with CUDA Support**
   
Based on your system's CUDA version, install PyTorch from [Pytorch.org](https://pytorch.org/). 
If unsure, you might need to install the [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) first.
	
6. **Install Required Python Packages**
While in the virtual environment, install all required packages:

`pip install -r requirements.txt`

> [!TIP]
> To re-enter the virtual environment in the future, run `activate_venv.bat`.

## Folder Structure

- **dataset**: For dataset images and annotations.
- **train**: Training data.
- **valid**: Validation data.
- **test**: Test data (optional).
- **generate_input**: Place images here for detection testing.
- **generate_output**: Where detection outputs and annotations are saved.
- **models**: For storing base and trained models.
- **training_output**: Contains trained models, curves, results, and tests.
	
## Scripts Overview

> **Important:** Activate the virtual environment before running any scripts.

- **train.py**: Configures and launches model training.
- **generate.py**: Runs inference to generate detection results.
- **yoloOutputCopyMatchingImages.py**: Aids in selecting and copying images based on matching names for further processing.
- **yoloOutputToYoloAnnotations.py**: Converts detection outputs into YOLO training annotation format.
- **CocoGetClasses.py**: Extracts class names from a COCO dataset for YOLO training.
- **cocoToYoloAnnotations.py**: Converts COCO annotations to YOLO format.

## Scripts In Detail

> [yoloOutputCopyMatchingImages.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/yoloOutputCopyMatchingImages.py):
> This script is a small tool to help you select and copy images from one folder, based on matching image names of another folder.
> Example:
>   - You have a folder with input images (original) to detect something from.
>   - You run a detection model, and get another folder with overlays showing the detection.
>   - You then run a tool like [img-txt viewer](https://github.com/Nenotriple/img-txt_viewer) to remove text-image pairs of failed detections.
>   - Now you have a folder with only successful detections (curated). Now is when this script comes in.
>   - You now run the tool and choose those directories, and it will copy any original images that matches the names of the images that are in the curated folder.
>   - You can now run the (yoloOutputToYoloAnnotations.py) script to convert the output yolo detection coordinates, to yolo training annotations.

> [yoloOutputToYoloAnnotations.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/yoloOutputToYoloAnnotations.py):
> This script converts output yolo detection text-files, into yolo training annotation files. It's used together with the (yoloOutputCopyMatchingImages.py) script to generate training data from successful detections.

> [dataset/CocoGetClasses.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/dataset/CocoGetClasses.py):
> Extracts the class names from a downloaded COCO format dataset and outputs them in the Yolo training format. Place the script in the same folder as `_annotations.coco.json` and run the script. _Can be run outside of the venv._

> [dataset/cocoToYoloAnnotations.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/dataset/CocoToYoloAnnotations.py):
> Converts annotations from the COCO format to the Yolo training format. Place the script in the same folder as `_annotations.coco.json` and run the script. _Can be run outside of the venv._


# Training a custom detection model

This section is a brief guide on preparing and training a custom detection model.

### 1. Downloading a Dataset

Download a dataset, like the watermark dataset from Roboflow, in the YOLOv8 format. Unpack and move the directories into the `/dataset/` folder.
- [Watermark dataset by MFW](https://universe.roboflow.com/mfw-feoki/w6_janf)

![Watermark dataset](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/c79a6e7c-3f21-421c-8876-03676918afb8)

### 2. Dataset Preparation

Place images in `/train/images` and annotations in `/train/labels`. For COCO datasets, use `CocoGetClasses.py` and `cocoToYoloAnnotations.py` for conversion.

### 3. Configuring data.yaml

Edit `data.yaml` to match your dataset's number of classes and class names.


For our watermarks dataset, it should be:
```
nc: 1 # Number of classes
names: ['watermark'] # Class names
```

> [!TIP]
> Sometimes the classes are listed on the dataset website, like the screenshot below.
> 
> ![image](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/cb85228e-6e17-4be2-aa05-8a89cf352dc5)
>
> If it is not, you can try downloading the COCO.JSON version of the dataset, and run the CocoGetClasses.py script from this repository to extract the values you need for the dataset.yaml.


### 4. Setting Up train.py
Customize training parameters in `train.py` such as folder name, starting model, epoch count.
```
"folder_name" is the output folder name inside the `training_output` directory.
"starting_model" is which model to use for your training. You can copy the standard yolov8 models from the list above. The Nano-model is the smallest, trains faster, but usually performs worse. And Xtra Large is the opposite. Use the -seg models if you have a segmentation dataset.
"epoch_count" how many versions do you wish to train. 50 may be a good starting point. For a small model, or a model with a very small dataset, you could set this to 500. The training will automatically stop if no improvement is made in 50 epochs.
```

### 5. Running train.py
Execute `train.py` to start the training process. Models and results will be saved in the `training_output` directory.

Hopefully, you should have something like this now:
![image](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/5cbc9997-5274-42ca-a193-57cd553e5a91)

If you need to cancel the training, you can just close the window or press `CTRL + C` to interrupt.

You can find test results and your models in the `training_output` directory.

The script will always save your latest model (last.pt) and the currently best performing model (best.pt), in the /training_output/project_name/weights/ directory.

### 6. Testing Your Model
After training, use `generate.py` with your model to detect classes on new images. Place test images in `/generate_input` and run the script to generate outputs in `/generate_output`.
Copy your output model into the `models` directory. Now is a good time to rename it to something suitable, like `watermarks_s_yolov8_v1.pt`.
> [!TIP]
> You may want to try both the `last.pt` and `best.pt` separately to see which model perfoms the best for you.

Open `generate.py` to edit some parameters.
```
"model_path" is the path to your model.
"mode" should be set to detection or segmentation based on what you want to output
"selected_classes" is a list of the classes you wish to identify and detect when running the script.
"class_overrides" is a list of overrides. Use this if you wish to substitute one class with another. This could be useful if the model is trained on the classes in the wrong order, or if you just wish to change the name of the label in the overlay images.
"confidence_threshold" is the detection confidence needed to make it consider it a positive detection.
```

Now place all images you wish to test your model on in the `/generate_input` folder.

While inside the environment, run `python generate.py` to launch the generation.

The output images with annotations overlay, as well as the detections text-files will be found in the `/generate_output` folder.

![image](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/6d3a2c3c-5c84-4970-b016-59c6394e219a)


# F.A.Q / Known Issues
### No images found in {img_path}
```
Traceback (most recent call last):
  File "[YOUR INSTALL PATH HERE]\ultralytics\ultralytics-venv\Lib\site-packages\ultralytics\data\base.py", line 119, in get_img_files
    assert im_files, f"{self.prefix}No images found in {img_path}"
AssertionError: train: No images found in D:\AI\Projects\Yolov8Tutorial\dataset\train
```
> [!WARNING]
> Ensure that your training data is available in the /dataset/train/ folder.

### NotImplementedError: Could not run 'torchvision::nms'
```
NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'torchvision::nms' is only available for these backends: [CPU, Meta, QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMeta, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
```
> [!CAUTION]
> It's likely a pytorch installation issue.
> 
> Activate the venv and run this command `pip list`.
> 
> Look for the torch, torchaudio and torchvision lines. They should say something like:
> 
> 	- torch              2.2.0+cu118
> 	- torchaudio         2.2.0+cu118
> 	- torchvision        0.17.0+cu118
> 	  
> If they don't have +cu118, your cuda installation is not working in the environment.
> 
> Make sure you follow the installation steps in the exact order. If they are done in the wrong order, you may not have a working environment.
> 
> You can try this command and see if it helps: `pip install torchvision --upgrade --force-reinstall`.
