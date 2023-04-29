# Generative-Adversarial-Training

This is the repository of Thomas Fiello's and Justus Renkhoff's class project for CS595A at Embry-Riddle Aeronautical University (Spring 2023).

## Getting Started

### Setting up environment
First, we recommend to create an environment to install the appropriate Python packages. For this we recommend to use conda (https://conda.io/projects/conda/en/latest/index.html). 

After installing Anaconda on your system (https://www.anaconda.com/download/), use the following command to create a conda environment with Python 3.10. 
```console
conda create python=3.10 --name <env_name>
``` 

This environment can then be activated using the command:
```console
conda activate <env_name>
``` 

Inside the repository there is a requirements.txt which contains all necessary packages to execute the code. These packages can be installed automatically by executing the following command: 
```console
pip install -r requirements.txt
``` 

Depending on the development environment used, further steps are required to add the environment as a kernel. In Visual Studio it should now be possible to select your Environment as Kernel. In Jupyter it might be needed to first install ipykernel using the command:
```console
conda install -c anaconda ipykernel
``` 
And then adding your Environment to Jupyter with:
```console
python -m ipykernel install --user --name=<env_name>
``` 

### Downloading required data



## Running the Code



## Disclaimer
The code and these instructions were tested using Python 3.10.11 and Windwos 11. Depending on your operating system modifications might be needed. Feel free to contact us in case any problems occure or questions arise.
