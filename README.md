# bachelor-goldenImprints

This repository contains the code for my bachelor's project from Aarhus University, fall 2023.

## Project Description
The project aims to examine representation learning for visual art by exploring and comparing which state of the art, pre-trained vision models best represent paintings. To do so, ... different classifiers have been constructed, which tries to classify features of an art dataset based of image embeddings extracted by pre-trained vision models from the *timm* library. 

## Project Structure

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```nbs```  | Notebooks used in the project         |
| ```out``` | Contains all outputs from running the scripts. See ... for more information.|
| ```src```  | Folder with scripts used for downloading the data, extracting features and building and fitting classifier models. See ... for more information.       |
| ```run.sh```    | Bash script for runing the entire analysis with predefined arguments  |
| ```setup.sh```  | Bash script for setting up virtual environment |
| ```requirements.txt```  | Packages needed to run the analysis|


## Technical Implementation
All code has been written and implemented in an Ubuntu 22.04.4 operating system using Python 3.10.12.

### Set up virtual environment
First, clone this repo with ```git clone```.

Make sure to have the *venv* package for Python installed. If not, run:

```
sudo apt-get update

sudo apt-get install python3-venv
```

Next, to create a virtual environment (```env```) and install required packages, run:

```
bash setup.sh
```




### Usage

### Runtime


