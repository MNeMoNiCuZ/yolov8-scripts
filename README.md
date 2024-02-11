# Yolov8 Training & Inference Scripts
This repository helps you train detection models, as well as use them to generate detection outputs (image and text).

> [!IMPORTANT]
> Requires **Python 3.11** or newer.

## Installation Instructions - Windows
1. Download or git clone this repository to any folder
`git clone https://github.com/MNeMoNiCuZ/yolov8-scripts`

2. Enter the folder
`cd yolov8-scripts`

3. Git Clone Ultralytics inside this folder
`git clone https://github.com/ultralytics/ultralytics`

4. Run `setup.bat`. It will ask you to enter an environment name. Press Enter to use the defaults. Only change it if you know what you are doing.
The venv should be created inside the Ultralytics folder. This will also create a few empty folders for you, and an environment activation script (`activate_venv.bat`). It should also activate the environment for you for the next step.

5. Install torch for your version of CUDA ([Pytorch.org](https://pytorch.org/)):
   
	`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

6. Inside the (venv), install requirements using `pip install -r requirements.txt`.

> [!TIP]
> In the future, you can enter the virtual environment by running `activate_venv.bat`.

## Folder structure information
- **dataset**: Contains your dataset images and annotation files (captions), in subdirectories.
   - **train**: Contains your training data.
   - **valid**: Contains your validation data (start with ~10% of your training data moved here).
   - **test**: Contains your test data (optional).
 - **generate_input**: Place the images to test your detection in here. _Note: These could also be placed inside dataset/test as long as you update the generate.py to that directory._
 - **generate_output**: Generated detections and output annotation files are placed here.
 - **models**: Downloaded base models are placed here. Place your models in here when generating.
 - **training_output**: This is where trained models (weights), curves, results and tests will end up after training.
	
## Scripts
> [!IMPORTANT]
> Remember to enter the environment to run the scripts.

> [train.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/train.py):
> The training script. Configure the training folder name, epoch count, batch size and starting model. It requires the dataset to be properly setup.

> [generate.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/generate.py):
> The inference script used to generate detection results. Configure the model name, which classes to detect, class overrides

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
This is not meant as a full-fledged guide, but a few guiding steps. For more information, read some [Yolo documentation](https://docs.ultralytics.com/datasets/detect/).

## 1. Downloading a dataset
- For our example, we will download [this watermark dataset](https://universe.roboflow.com/mfw-feoki/w6_janf) from [Roboflow.com]([https://universe.roboflow.com](https://universe.roboflow.com/mfw-feoki/w6_janf)) by user [MFW](https://universe.roboflow.com/mfw-feoki).
> [!TIP]
> Download the dataset in the Yolov8 format.
![image](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/c79a6e7c-3f21-421c-8876-03676918afb8)

- Unpack the file and move the train/test/valid-directories into the /dataset/ folder for your project.

![image](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/5c72e5f8-531c-4668-918d-bf9c6f925831)


## 2. Dataset preparation
If you downloaded a Yolov8 dataset, everything should be fine already. Images are placed in `/train/images`, and the annotations are placed in `/train/labels`.
> [!TIP]
> You can also have both the images and annotations right inside the root of the `/train` folder without any /images and /labels subfolders. The same goes for the valid and test folders.

If you downloaded a COCO dataset, you can use the [dataset/CocoGetClasses.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/dataset/CocoGetClasses.py) and [cocoToYoloAnnotations.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/dataset/CocoToYoloAnnotations.py) scripts to convert the dataset to yolov8. There's also a built-in command that I didn't know about when I wrote the scripts.

## 3. Data.yaml configuration
The [dataset/data.yaml](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/dataset/data.yaml) must be configured for your dataset.

Edit the file and make sure that the number of classes matches the number of classes of your dataset, as well as the list of class names.

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


## 4. Configure train.py
Open `train.py` and edit some parameters.
```
"folder_name" is the output folder name inside the `training_output` directory.
"starting_model" is which model to use for your training. You can copy the standard yolov8 models from the list above. The Nano-model is the smallest, trains faster, but usually performs worse. And Xtra Large is the opposite.
"epoch_count" how many versions do you wish to train. 50 may be a good starting point. For a small model, or a model with a very small dataset, you could set this to 500. The training will automatically stop if no improvement is made in 50 epochs.
```

## 5. Run train.py
While inside the environment, run `python train.py` to launch the training.

Hopefully, you should have something like this now:
![image](https://github.com/MNeMoNiCuZ/yolov8-scripts/assets/60541708/5cbc9997-5274-42ca-a193-57cd553e5a91)

If you need to cancel the training, you can just close the window or press `CTRL + C` to interrupt.

You can find test results and your models in the `training_output` directory.

The script will always save your latest model (last.pt) and the currently best performing model (best.pt), in the /training_output/project_name/weights/ directory.

## 6. Generate / Detect / Test your model
Copy your output model into the `models` directory, you can also rename it to something suitable, like `watermarks_s_yolov8_v1.pt`.
> [!TIP]
> You may want to try both the `last.pt` and `best.pt` separately to see which model perfoms the best for you.

Open `generate.py` to edit some parameters.
```
"model_path" is the path to your model.
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
