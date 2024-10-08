# ProBID-NET is a deep-learning model for designing amino acid on protein-protein interfaces.

## Installation (tested on Linux)
### 1. download the source code 
   run `git clone https://github.com/ComputArtCMCG/ProBID-NET.git` to clone the repository. \
   **Alternatively,**
   download the zip file, unzip it and and navigate to the folder using the following commands:\
   `unzip ProBID-NET-main.zip && cd ./ProBID-NET-main/`
   
### 2. download the checkpoint of the model
   The checkpoint of model trained on both Chain-chain interface and domain-domain interface sets is available at    https://figshare.com/s/ebbd5184c0a46fb2b179, download *modeloutput0.hdf5* and move it to the *model* directory in *ProBID-NET-main*.

### 3. prepare anaconda environment
   Install anaconda from https://anaconda.org/ if it is not installed in the system.\
   Run the following commands to setup the env:\
   `conda create -n keras2.8.0 python=3.9`\
   `conda activate keras2.8.0`\
   `conda install keras=2.8.0`\
   `pip install pandas biopython`

### 4. make prediction
   Run `bash ./ProBID-Net_run.sh examples/1euv.input ./examples/` to make prediction on an test protein.\
   The prediction output is saved in *examples/output_pred*.\
   The first option of ProBID-Net_run.sh is a file containing list of PDB file and chain ID to predict.\
   The second option points to a directory where PDB files are saved.


## Dataset
The lists of protein-protein complex structures for the training set and  test sets are available in the *dataset* folder.


## Reference
Zhihang Chen et. al, ProBID-Net: A Deep Learning Model for Protein-Protein Binding Interface Design, *submitted*  


   

If you encounter any issues during installation, please open an issue.


