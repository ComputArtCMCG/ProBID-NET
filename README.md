The checkpoint of model trained on both Chain-chain interface and domain-domain interface sets is available at https://figshare.com/s/ebbd5184c0a46fb2b179; DOI 10.6084/m9.figshare.25333570.


After downloading the checkpoint, you can run ProBID-Net by following the steps:

1. Create conda environment and install keras
conda env create -n ProBID -f environment.yaml

2. Download the checkpoint of model and save it in the 'models' fold
   
3. Prepare a list file that contains the prefix of the PDB files of protein-protein complexes and
 the chain to predict separated by '  '(i.e., '1euv.pdb  B' )

4. In the ProBID-Net_run.sh script, change 'ProBID_path=./' to either the actual path of the ProBID-Net main folder
   or leave it as the current setting if you are running the script from the ProBID-Net directory

5. Activate the ProBID env if it is not active and make predictions with
./ProBID-Net_run.sh [pdblist] [pdbpath]

where [pdblist] is the file in step 2, and [pdbpath] is the path that all PDB files are stored.

6.You can find a total of six output predictions in the 'output_pred' folder,where 'xxxx.pred_0' is 
predicted by the model with the domain-domain interface added as the training set.

Example:

If you encounter any issues during installation, please raise a question in the issue section of our repository.


