# Cancer Instance Segmentation

Cell Nuclei Segmentation on Cancer Instance Segmentation (PanNuke) Using Python and PyTorch.


Overview
------------
This Jupyter notebook implements the segmentation of nuclei in whole-slide images (WSIs) of cancerous tissue slides using the PanNuke dataset. It is designed to work with the PanNuke extension dataset which includes nearly 200,000 nuclei across various cancer types. The notebook contains detailed Pytorch code for applying deep learning models to perform nuanced segmentation tasks that are crucial for advancing computational pathology.


Dataset
------------
[Link](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification/data)


Results
------------
These figures show some representative input images and corresponding ground truths and predicted images:

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%201.png?raw=true)

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%202.png?raw=true)

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%203.png?raw=true)

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%204.png?raw=true)




Dependencies
------------
- PyTorch >= 2.2.1+cu118
- CUDA >= 11.8
- Python >= 3.9
