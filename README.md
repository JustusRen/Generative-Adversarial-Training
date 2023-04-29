# Generative-Adversarial-Training

This is the repository of Thomas Fiello's and Justus Renkhoff's class project for CS595A at Embry-Riddle Aeronautical University (Spring 2023).

## Getting Started

### Setting up environment
First, we recommend to create an environment to install the appropriate Python packages. For this we recommend to use conda (https://conda.io/projects/conda/en/latest/index.html). 

After installing Anaconda on your system (https://www.anaconda.com/download/), you can create a conda environment. Inside the repository is a environment.yml file which contains all necessary packages to execute the code. The environment including the packages can be created and installed by executing the following command: 
```console
conda env create -f environment.yml
``` 

The name of created environment <env_name> should be env_gen_ai_cs595. This environment can then be activated using the following command. 
```console
conda activate <env_name>
``` 

Depending on the IDE used, further steps may be required to add the environment as a kernel. n Jupyter it might be needed to add your environment to Jupyter with:
```console
python -m ipykernel install --user --name=<env_name>
``` 

### Downloading required data

To use the code, some test data is needed. This can be downloaded from the following link. The files also include two pretrained models:
https://myerauedu-my.sharepoint.com/:f:/g/personal/renkhofj_my_erau_edu/EmXD6hjQIfRDvVXuyDKWkNsBeutqi0PTNYQP7e6Trg7rSw?e=I6gAlf

The full data set can be downloaded here:
https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

## Running the Code

To run the code, simply add the downloaded folder into the repository. And specify the path in the corresponding notebook, depending on your folder name. If you use the test_data folder, you don't need to anything. You can simply extract the data and copy the unzip folder into the repository.

Execute the notebooks in the following order:
1.) Run preprocess.ipynb
2.) Run train_classification_model.ipynb
3.) Run FGSM_adversarial_attack.ipynb
4.) Run gan_comparison.ipynb

## Disclaimer
The code and these instructions were tested using Python 3.10.11 and Windwos 11. Depending on your operating system modifications might be needed. Feel free to contact us in case any problems occure or questions arise.
