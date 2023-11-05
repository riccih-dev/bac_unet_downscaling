## Temperature based Downscaling
This repository contains the machine learning code used in the paper: Temperature based Downscaling using U-Nets and Standardized Anomalies

## Installation
The project requires the following libaries:

Before running the experiment, these libaries have to be installed.

```bash
python setup.py install
```


## Running the experiment
The following script "downscaling_experiement" will running the experiments mentioned in the paper. This includes loading the data, training the model and testing it on unseen temperature data. After runnning the results will be visibile in the folder "visualizations". 

```bash
ipython -c "%run <notebook>.ipynb"
```

