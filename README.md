# Cancer Instance Segmentation

Cell Nuclei Instance Segmentation on Cancer Instance Segmentation (PanNuke) Using Python and PyTorch.


Overview
------------
This Jupyter notebook implements the segmentation of nuclei in whole-slide images (WSIs) of cancerous tissue slides using the PanNuke dataset. It is designed to work with the PanNuke extension dataset which includes nearly 200,000 nuclei across various cancer types. The notebook contains detailed Pytorch code for applying deep learning models to perform nuanced segmentation tasks that are crucial for advancing computational pathology.


Dataset
------------
[Link](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)

IoU Loss Function
------------
The IoU Loss function is tailored for the instance segmentation of cell nuclei by computing the IoU metric as a loss value. The implementation supports weighting for class balance, and is designed to handle multi-class segmentation tasks efficiently.


Results
------------
These figures illustrate representative input images, along with their corresponding ground truths and predicted images:

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%201.png?raw=true)

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%202.png?raw=true)

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%203.png?raw=true)

![Sample](https://github.com/mansour2002/Digital-Pathology-Segmentation/blob/main/Figures/Nuclei%20Segmentation%204.png?raw=true)




Dependencies
------------
- PyTorch >= 2.2.1+cu118
- CUDA >= 11.8
- Python >= 3.9
