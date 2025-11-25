#!/bin/bash
# Create project directory
mkdir coco_evaluation_project
cd coco_evaluation_project

# Download validation set
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip val2017.zip
unzip annotations_trainval2017.zip

# Clean up
rm *.zip

echo "COCO dataset ready!"