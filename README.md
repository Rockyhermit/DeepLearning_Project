# AI-Powered Self-Checkout System – Apple Detection with YOLO

This repository contains the code, models, and documentation for a mobile self-checkout application focused on detecting apples using computer vision techniques. The project explores multiple object detection architectures, ultimately developing a custom YOLO model optimized for detecting apples under challenging conditions such as occlusion and small object scale.

## Project Objectives

- Enable real-time apple detection for self-checkout scenarios.
- Address occlusion and scale issues with custom architecture and post-processing techniques.
- Evaluate multiple models and select the most suitable approach based on performance and real-time feasibility.

## Datasets Used

- **COCO Dataset (Class 47 - Apple)**  
  Source: [https://cocodataset.org/#home](https://cocodataset.org/#home)  
  Used for training object detection models. Only the apple class (class ID 47) was extracted.

- **MinneApple Dataset**  
  Source: [https://rsn.umn.edu/projects/orchard-monitoring/minneapple](https://rsn.umn.edu/projects/orchard-monitoring/minneapple)  
  Used for analyzing detection performance on small, distant apples and conducting resolution analysis.

## Models Evaluated

- SSD (Single Shot Detector)
- Faster R-CNN
- YOLO (You Only Look Once)
  - Custom YOLO architecture designed for improved apple detection.
  - Pretrained YOLOv11x evaluated with post-processing techniques.

## Technical Highlights

- Composite Loss Function: CIoU, Focal Loss, and Distribution Focal Loss
- Optimizer: AdamW with cosine learning rate decay and occlusion-aware scheduling
- Post-processing: Multi-scale inference, Soft-NMS, sliding window detection
- Evaluation Metrics: Precision, Recall, mAP@50, mAP@50:95, box loss, classification loss, DFL loss
- The final weight for occlusion best.pt --> https://www.mediafire.com/file/m5b2klky60na8dn/best.pt/file

## Results

- YOLO outperformed SSD and Faster R-CNN in both accuracy and real-time capability.
- Resolution analysis on the MinneApple dataset provided empirical recommendations for detecting apples at varying distances.

## Team Members

- Pratheek Tirunagari
- Ashruj Gautam

## Documentation

The full technical report is available in `Report.pdf`.
