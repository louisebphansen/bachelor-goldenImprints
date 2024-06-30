# Representing Visual Art

This repository contains the code for the paper on representing visual artworks for the CHR conference in Aarhus, 2024-

## Project Description
The paper compares the representational abilities of unimodal and multimodal state-of-the-art pre-trained vision models from the *timm* library, by employing their embeddings in three domain-specific downstream tasks: genre classification, style classification, and artist classification in the WikiArt dataset.


## Project Structure

| <div style="width:120px"></div>| Description |
|---------|:-----------|
| ```nbs```  | Notebooks used for creating plots        |
| ```out``` | Contains all outputs from running the scripts|
| ```src```  | Folder with scripts used for downloading the data, extracting features and building and fitting classifier models       |
| ```run.sh```    | Bash script for running the entire analysis with predefined arguments  |
| ```setup.sh```  | Bash script for setting up virtual environment |
| ```requirements.txt```  | Packages required to run the code|


## Technical Implementation
All code has been written on an Ubuntu 22.04.4 operating system using Python 3.10.12. It can therefore not be guaranteed that it will work on other operating systems.

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

This first downloads the WikiArt dataset from HuggingFace (https://huggingface.co/datasets/huggan/wikiart). Next,  different pretrained models from the ```timm``` library are used to extract embeddings from the dataset. Finally, seperate classification models are fit for each models for the three features of the WikiArt dataset, i.e., genre, style and artist. 

**NB: Due to the size of the dataset and the number of models, running the entire analysis will take several days, depending on your computation power.
