#!/bin/bash
ProBID_path=./
fpdblist=$1
pdbdir=$2
#chain=$3

nnmodelpt=$ProBID_path/models
spt=$ProBID_path/scripts
cleaned=$ProBID_path/cleaned_pdb


while read i j 
do
	python $spt/clean_pdb_CNN.py $pdbdir/$i $cleaned/$i
	python $spt/split.py $cleaned/$i 
	python $spt/mask.py $cleaned/$i $cleaned/$i.$j 20 1 2 $spt/atom.channelID 
	mv $cleaned/$i.$j.hdf5 $ProBID_path/input_masked/
done<$1

nnmodel=""
for i in `seq 0 5`
do
nnmodel=$nnmodel" -model ${nnmodelpt}/modeloutput$i.hdf5"
done
python $spt/ProBID_pred.py -predict $fpdblist $nnmodel -datapath $ProBID_path/input_masked -outpath $ProBID_path/output_pred/ -batchsize 8
