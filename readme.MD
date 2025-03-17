# Skin Disease Classification - README
This project is aimed at building an efficient skin disease classification system using state-of-the-art computer vision models and deploying them as a real-time interactive application.

### Dataset Information:

DermNet (primary dataset): https://www.kaggle.com/datasets/shubhamgoel27/dermnet  or https://huggingface.co/datasets/Muzmmillcoste/dermnet

Other datasets explored : 

- SCIN (available on public Google cloud storage) gs://dx-scin-public-data/dataset/

- HAM10000 - https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification

### Additional Approaches

SCIN+HAM Folder: Contains a notebook with exploratory data analysis (EDA) and model building using SCIN and HAM10000 datasets. However, due to low accuracy and inconsistencies in image patterns, we decided to stick with the DermNet dataset for classifying major skin diseases.

Image Clustering Approach: We also attempted combining DermNet, SCIN, and HAM10000 datasets and clustered image labels. However, the classification quality did not meet our expectations. A corresponding notebook for this approach is also included.

### Main Solution

#### EDA & Model Development Notebooks
The Code Folder contains:

- DermNet_EDA_and_Models_part1.ipynb: Initial exploratory data analysis and model trials. 

- MobileNetv2.ipynb, finetuned_dinov2.ipynb (DINOv2) and YOLOv8.ipynb (YOLOv8) contains the models evaluated by training, validation, and test set accuracy. 

- VITfinetuned_dermnet.ipynb: Optimized fine-tuned Google ViT Transformer model using Optuna and hugging face image transformers. The final fine-tuned and optimized ViT Transformer model is saved as model.onnx. 

#### Inference & API Integration: 
The model is deployed using a Streamlit app (app.py).

#### To run the Streamlit app:

1. Download the model_checkpoint release file (model.onnx)
2. Open the folder "DermNet" in VS Code.
3. Go to terminal and "cd DermNet"
4. Run command "streamlit run app.py"
5. Upload an image from the Test Folder or capture a live image.
6. The prediction result will be displayed along with the model's confidence score.

A sample image (sample.jpg) is used for testing. The prediction on a sample image is captured in Prediction_on_sample_image.png.

