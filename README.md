# bachelor-goldenImprints

This repository contains the code for my bachelor's project from Aarhus University, January 2024.

## Project Description
The project aims to examine representation learning for visual art by exploring and comparing which state of the art, pre-trained vision models best represent paintings. To do so, ... different classifiers have been constructed, which tries to classify features of an art dataset based of image embeddings extracted by pre-trained vision models from the *timm* library. 

## Project Structure

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```nbs```  | Notebooks used in the project         |
| ```out``` | Contains all outputs from running the scripts. See ... for more information.|
| ```src```  | Folder with scripts used for downloading the data, extracting features and building and fitting classifier models. See ... for more information.       |
| ```run.sh```    | Bash script for running the entire analysis with predefined arguments  |
| ```setup.sh```  | Bash script for setting up virtual environment |
| ```requirements.txt```  | Packages needed to run the analysis|


## Technical Implementation
All code has been written and implemented on an Ubuntu 22.04.4 operating system using Python 3.10.12. Therefore, it cannot be guaranteed that it will work on other operating systems.

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
To run the entire analysis with predefined arguments, run:

``` 
bash run.sh
```

This uses three different pretrained models from the ```timm`````` library, ...., to extract embeddings from the WikiArt dataset. Next, (number) seperate classification models are fit for the three features of the WikiArt dataset, i.e., genre, style and artist. 

**NB: Due to the size of the dataset and the number of models, running the entire analysis will take several days or even weeks, depending on your computation power. See ```Runtime``` for more information on this.** 

### Runtime
All code was run on 1 NVIDIA V100 GPU with 45GB memory and 20 vCPU (Intel Zeon Gold 6230) on the UCloud platform.

#### Feature Extraction

| <div style="width:120px"></div>|  Description |
|---------|:-----------|
| ```nbs```  | Notebooks used in the project         |
| ```out``` | Contains all outputs from running the scripts. See ... for more information.|

