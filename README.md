The checkpoint of model trained on both Chain-chain interface and domain-domain interface sets is available at https://figshare.com/s/ebbd5184c0a46fb2b179; DOI 10.6084/m9.figshare.25333570.

After downloading the checkpoint, you can run ProBID-Net by following the steps:

1. Create conda environment and install keras
conda env create -n ProBID -f environment.yaml

2. Prepare a list file that contains the prefix of the PDB files of protein-protein complexes and t
he chain to predict separated by '  '(i.e., '1euv.pdb  B' )

3. In run.sh, change 'ProBID_path=./' to the actual path of the ProBID folder

4. Activate the ProBID env if it is not active and make predictions with
./run.sh [pdblist] [pdbpath]

where [pdblist] is the file in step 2, and [pdbpath] is the path that all PDB files are stored.

5.You can find a total of six output predictions in the 'output_pred' folder,where 'xxxx.pred_0' is 
predicted by the model with the domain-domain interface added as the training set.


