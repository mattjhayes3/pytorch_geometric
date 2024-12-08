#!/bin/sh
echo "command	val_loss_mean	val_loss_std	val_acc_mean	val_acc_std	test_acc_mean	test_acc_std	duration_mean	duration_std	runs"
#echo "Cora"
python gat.py --dataset=Cora --non_interactive
python gat.py --dataset=Cora --v2 --non_interactive
python gat.py --dataset=Cora --random_splits --non_interactive
python gat.py --dataset=Cora --v2 --random_splits --non_interactive
# echo "CiteSeer"
python gat.py --dataset=CiteSeer --non_interactive
python gat.py --dataset=CiteSeer --v2 --non_interactive
python gat.py --dataset=CiteSeer --random_splits --non_interactive
python gat.py --dataset=CiteSeer --v2 --random_splits --non_interactive
# echo "PubMed"
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --non_interactive
python gat.py --dataset=PubMed --v2 --lr=0.01 --weight_decay=0.001 --output_heads=8 --non_interactive
python gat.py --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits --non_interactive
python gat.py --dataset=PubMed --v2 --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits --non_interactive
# echo "Arxiv"
python gat.py --dataset=Arxiv --runs=20 --non_interactive --batch_norm --no_normalize_features
python gat.py --dataset=Arxiv --runs=20 --v2 --non_interactive --batch_norm --no_normalize_features
