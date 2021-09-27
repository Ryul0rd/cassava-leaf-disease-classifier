# cassava-leaf-disease-classifier

## Summary
Late submission to Kaggle's Cassava Leaf Disease Classification competition. Uses a ViT-based model for image classification.

## Usage
Before running anything: 
1) Download the cassava leaf disease dataset from https://www.kaggle.com/c/cassava-leaf-disease-classification/data
2) Extract Label_num_to_disease_map.json, train.csv, and the entire train_images folder to cassava-leaf-disease-classifier/data
3) Run the script make_image_folders.py

To train simply run train.py. A Weights and Biases account is necessary in its current form.