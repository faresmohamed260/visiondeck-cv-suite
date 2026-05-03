# Standalone Object Detection Assignment

This folder is independent from the rest of the repository.

It is set up for a separate coursework notebook that:

- prepares a small 3-class object detection dataset
- fine-tunes a pretrained YOLOv8 model
- evaluates the trained model
- reports the results inside a notebook

Chosen classes:

- `person`
- `bottle`
- `cell phone`

The dataset is built from COCO 2017 validation annotations and images, then converted into YOLO format for fine-tuning.
