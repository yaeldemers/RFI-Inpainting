# RFI-Inpainting-with-DSS-Layers
## Overview
RFI-Inpainting-with-DSS-Layers is a machine learning project focused on addressing the challenges of radio frequency interference (RFI) in ground-based radio photon measurements. The project leverages convolutional neural networks (CNNs) to perform inpainting, effectively overcoming data-flagging issues caused by RFIs during measurements conducted by the HERA (Hydrogen Epoch of Reionization Array) instrument.

## Project Description
The main goal of this project is to develop an improved in-painting technique using machine learning, specifically by modifying the CNN model proposed by Dr. Michael Pagano. The uniqueness of the HERA instrument lies in its upward-facing design, which results in some symmetry in the measured data. By exploiting the daily repetition of measurements for the same point in the sky, we aim to enhance the inpainting behavior.

## Key Objectives
- Mitigate the impact of data flagging due to RFIs on radio photon measurements.
- Utilize deep learning techniques, including convolutional neural networks, to perform inpainting.
- Incorporate Deep Sets for Symmetric elements (DSS) layers to exploit the symmetry observed in HERA's measurements.

## Project Structure
The project is structured as follows:

- **data**: Contains the preprocessed datasets used for training, validation, and testing. The subdirectories include "train," "val," and "test," each with "images" and "masks" folders.
- **models**: Contains the implementation of the U-Net model with DSS layers (model_upaint.py) and their respective checkpoints for the trained models.
- **notebooks**: Contains the data vizualizatoin notebook (data_processing.ipynb) for data interactive visualization of the predictions.
- **scripts**: Contains the main scripts used for training (train.py), prediction (predict.py), and visualization (visualize.py).
- **utils**: Contains utility functions used in the project, including data manipulation, visualization, and custom loss functions.
- **figures**: Stores generated figures and plots.
- **outputs**: Contains the output files, including data_out.npz and logs for statistics of training (model_log.csv).

# Getting Started
1. Clone the repository to your local machine.
2. Set up the required environment with the necessary dependencies, including TensorFlow and Keras.
3. Run the train.py script to train the U-Net model with DSS layers using the prepared datasets.
4. Use the predict.py script to make predictions on new data or test sets.
5. Visualize the predictions and generate figures using the visualize.py script.
Refer to the logs in the "outputs" directory for training statistics and performance evaluation.

## Contributors
- Dr. Michael Pagano (michael.pagano@mail.mcgill.ca): Original model implementation and guidance.
- Yael-Jeremy Demers (yael.demers@mail.mcgill.ca): Exploration of the integration of Deep Sets for Symmetric elements layers (DDS) to address radio frequency interference challenges.

## Acknowledgments
Special thanks to Dr. Michael Pagano for his valuable contributions and guidance in this project. I would also like to express my gratitude to Prof. Adrian Liu as his expertise and insightful feedback have greatly contributed to the success of this work.

_Notes: The dataset used in this project is not publicly available due to data sensitivity and usage restrictions. If you are interested in collaborating on this research or would like access to the dataset for academic purposes, please contact Prof. Adrian Liu at acliu@physics.mcgill.ca._