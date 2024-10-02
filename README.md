The checkpoint of model trained on both Chain-chain interface and domain-domain interface sets is available at https://figshare.com/s/ebbd5184c0a46fb2b179; DOI 10.6084/m9.figshare.25333570.
The lists of protein-protein complex structures for the training set, the added training set, and all three test sets are available in the following files:

'Chain-chain_interface_train_set.txt'
'Added_domain-domain_interface_set.txt'
'All_3_test_set_list.txt'

After downloading the checkpoint, you can run ProBID-Net by following the steps:

1. Unzip the downloaded file and navigate to the folder
   unzip ProBID-NET-main.zip
   cd ./ProBID-NET-main/

2. Create conda environment and install keras
  conda env create -n ProBID -f environment.yaml

3. Download the checkpoint of model and save it in the 'models' fold
   
4. Prepare a list file that contains the prefix of the PDB files of protein-protein complexes and
 the chain to predict separated by '  '(i.e., '1euv.pdb  B' )

5. In the ProBID-Net_run.sh script, change 'ProBID_path=./' to either the actual path of the ProBID-Net main folder
   or leave it as the current setting if you are running the script from the ProBID-Net directory

6. Activate the ProBID env if it is not active and make predictions with
  ./ProBID-Net_run.sh [pdblist] [pdbpath]
   where [pdblist] is the file in step 2, and [pdbpath] is the path that all PDB files are stored.

7.You can find a total of six output predictions in the 'output_pred' folder,where 'xxxx.pred_0' is 
  predicted by the model with the domain-domain interface added as the training set.

Example:
In the 'Example' folder, we demonstrate how to predict the amino acids on the interface residues of Chain B in the '1euv.pdb' complex structure.
First, the '1euv.pdb' file was split into two separate structures: '1euv.pdb.A' and '1euv.pdb.B'. In the '1euv.pdb.B' file, the amino acid types 
at the interface residues of Chain B were masked as 'XXX'.
Next, the structural information from '1euv.pdb.B' was encoded into an HDF5 file named '1euv.pdb.B.hdf5', which was used as the input for ProBID-Net.
After running the prediction, the probabilities for each of the 20 natural amino acids were output and saved in the file '1euv.pdb.B.pred'.

If you encounter any issues during installation, please raise a question in the issue section of our repository.


