# Yolov8 Training & Inference Scripts

## Installation Instructions - Windows
1. Download or git clone this repository to any folder
`git clone https://github.com/MNeMoNiCuZ/yolov8-scripts`

2. Enter the folder
`cd yolov8-scripts`

3. Git Clone Ultralytics inside this folder
`git clone https://github.com/ultralytics/ultralytics`

4. Run setup.bat. It will ask you to enter an environment name. Press Enter to use the defaults. Only change it if you know what you are doing.
The venv should be created inside the Ultralytics folder. This will also create a few empty folders for you, and an environment activation script (`activate_venv.bat`). It should also activate the environment for you for the next step.

5. Inside the (venv), install requirements using `pip install -r requirements.txt`.

6. Install torch for your version of CUDA [Pytorch.org](https://pytorch.org/) `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

The setup should now be done.

> [!TIP]
> In the future, you can enter the virtual environment by running activate_venv.bat.

> [!IMPORTANT]
> From inside the environment, you can run the [train.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/train.py) or [generate.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/generate.py) to train or generate.

## Installation Instructions - Other systems
> [!CAUTION]
> I have no idea. If someone does it, feel free to update this section.

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
[train.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/train.py):

The training script. It has a few parameters to tweak like the training folder name, epoch count, batch size and starting model. It requires the dataset to be properly setup.

[generate.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/generate.py):

The inference script used to generate detection results. Configure the model name, which classes to detect, class overrides

[yoloOutputCopyMatchingImages.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/yoloOutputCopyMatchingImages.py):

This script is a small tool to help you select and copy images from one folder, based on matching image names of another folder.

Example:
  - You have a folder with input images (original) to detect something from.
  - You run a detection model, and get another folder with overlays showing the detection.
  - You then run a tool like [img-txt viewer](https://github.com/Nenotriple/img-txt_viewer) to remove text-image pairs of failed detections.
  - Now you have a folder with only successful detections (curated). Now is when this script comes in.
  - You now run the tool and choose those directories, and it will copy any original images that matches the names of the images that are in the curated folder.
  - You can now run the (yoloOutputToYoloAnnotations.py) script to convert the output yolo detection coordinates, to yolo training annotations.

[yoloOutputToYoloAnnotations.py](https://github.com/MNeMoNiCuZ/yolov8-scripts/blob/main/yoloOutputToYoloAnnotations.py):

This script converts output yolo detection text-files, into yolo training annotation files. It's used together with the (yoloOutputCopyMatchingImages.py) script to generate training data from successful detections.

# Training a custom detection model
## Dataset preparation
dataset.yaml must be configured for your dataset.

Todo: Write guide

