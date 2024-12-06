#!/bin/sh

#echo "Cora"
python gat.py --dataset=Cora
python gat.py --dataset=Cora --v2
python gat.py --dataset=Cora --random_splits
python gat.py --dataset=Cora --v2 --random_splits
# echo "CiteSeer"
python gat.py --dataset=CiteSeer
python gat.py --dataset=CiteSeer --v2
python gat.py --dataset=CiteSeer --random_splits
python gat.py --dataset=CiteSeer --v2 --random_splits
# echo "PubMed"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8
python gat.py --dataset=PubMed --v2 --lr=0.01 --weight_decay=0.001 --output_heads=8
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits
python gat.py --dataset=PubMed --v2 --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits
