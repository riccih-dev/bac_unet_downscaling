# Temperature-Based Downscaling with U-Net and Standardized Anomalies

This repository contains the implementation for my bachelor's thesis project, focusing on temperature-based downscaling using a U-Net model in combination with standardized anomalies. The project is developed entirely in Python.

## Overview

The objective of this research is to implement a downscaling technique that utilizes the power of U-Net, a convolutional neural network model known for its effectiveness in image segmentation, and leverages standardized anomalies to refine temperature predictions at a finer scale.

## Features

- **U-Net Model Implementation:** The core of the project involves the implementation of a U-Net architecture specifically tailored for temperature downscaling.
- **Standardized Anomalies Integration:** Utilizing standardized anomalies in conjunction with the U-Net model to enhance the downscaling process.
- **Python Implementation:** The entire project is coded in Python, utilizing libraries such as TensorFlow, Keras, and other data manipulation tools.

## Repository Structure

The repository includes the Python scripts and Jupyter notebooks used for the implementation. The codebase is organized as follows:

- `model/`: Implementation of the U-Net architecture.
- `utility/`: 

### Data Directory
The data directory is excluded from version control as it contains the datasets utilized in the downscaling process. Due to the large size or dynamic nature of the data, it is not stored in the Git repository to avoid unnecessary bloating of the repository size. The required data will be automatically downloaded when running the downscaling process script. The process will fetch the necessary datasets and place them in this directory for analysis and downscaling using the implemented techniques.


## Requirements

To run the code from this repository, ensure you have the required libraries installed. You can install them using the provided requirements.txt file. To install the necessary libraries, run the following command:

```
pip install -r requirements.txt
```

## Running the experiment
The following script "downscaling_experiement" will running the experiments mentioned in the paper. This includes loading the data, training the model and testing it on unseen temperature data. After runnning the results will be visibile in the folder "visualizations". 

```bash
ipython -c "%run <notebook>.ipynb"
