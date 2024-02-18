# Imports
import os
import torch
from ultralytics import YOLO

def main():
    # Check if CUDA (GPU support) is available
    training_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Using device:", training_device)
    # Load a pretrained model
    # Model Options:
    '''
    yolov8n.pt # Nano Detection
    yolov8s.pt # Small Detection
    yolov8m.pt # Medium Detection
    yolov8l.pt # Large Detection
    yolov8x.pt # Xtra Large Detection

    yolov8n-seg # Nano Segmentation
    yolov8s-seg # Small Segmentation
    yolov8m-seg # Medium Segmentation
    yolov8l-seg # Large Segmentation
    yolov8x-seg # Xtra Large Segmentation
    '''
    # User settings
    output_dir = 'training_output'
    folder_name = 'watermark'
    starting_model = 'models/yolov8s.pt' # Choose the model size from the list above, will be downloaded
    batch_size = 32 # Batch size for training
    epoch_count = 50 # Number of training epochs

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Absolute path to dataset.yaml
    dataset_path = os.path.abspath('dataset/data.yaml')

    # Run the training
    modelYolo = YOLO(starting_model)
    modelYolo.train(data=dataset_path, epochs=epoch_count, batch=batch_size, device=training_device, project=output_dir, name=folder_name)

    # Evaluate model performance on the validation set
    metrics = modelYolo.val()

    # Optional: Export the model to alternative formats
    # Format Options:
    '''
    Format      	Argument        Model 	                Metadata 	Arguments
    PyTorch 	    - 	            yolov8n.pt 	            yes 	    -
    TorchScript 	torchscript 	yolov8n.torchscript 	yes	        imgsz, optimize
    ONNX 	        onnx 	        yolov8n.onnx 	        yes 	    imgsz, half, dynamic, simplify, opset
    OpenVINO 	    openvino 	    yolov8n_openvino_model/ yes 	    imgsz, half, int8
    TensorRT 	    engine 	        yolov8n.engine 	        yes 	    imgsz, half, dynamic, simplify, workspace
    CoreML 	        coreml 	        yolov8n.mlpackage 	    yes 	    imgsz, half, int8, nms
    TF SavedModel 	saved_model 	yolov8n_saved_model/ 	yes 	    imgsz, keras, int8
    TF GraphDef 	pb 	            yolov8n.pb 	            no 	        imgsz
    TF Lite 	    tflite 	        yolov8n.tflite 	        yes 	    imgsz, half, int8
    TF  Edge TPU 	edgetpu 	    yolov8n_edgetpu.tflite 	yes 	    imgsz
    TF.js 	        tfjs 	        yolov8n_web_model/ 	    yes 	    imgsz, half, int8
    PaddlePaddle 	paddle 	        yolov8n_paddle_model/ 	yes 	    imgsz
    ncnn 	        ncnn 	        yolov8n_ncnn_model/ 	yes 	    imgsz, half
    '''
    # path = model.export(format="onnx") # Export to alternative formats

    # Keep the script running (Optional)
    input("Press Enter to exit...")
if __name__ == '__main__':
    main()
