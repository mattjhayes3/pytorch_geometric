# #!/bin/sh
echo "command	val_loss_mean	val_loss_std	val_acc_mean	val_acc_std	test_acc_mean	test_acc_std	duration_mean	duration_std	runs"

# # echo "Cora"
# # echo "===="

# echo "GCN"
python gcn.py --loss=loge --dataset=Cora
python gcn.py --loss=loge --dataset=Cora --random_splits

# echo "GAT"
python gat.py --loss=loge --dataset=Cora
python gat.py --loss=loge --dataset=Cora --random_splits

# echo "Cheby"
python cheb.py --loss=loge --dataset=Cora --num_hops=3
python cheb.py --loss=loge --dataset=Cora --num_hops=3 --random_splits

# echo "SGC"
python sgc.py --loss=loge --dataset=Cora --K=3 --weight_decay=0.0005
python sgc.py --loss=loge --dataset=Cora --K=3 --weight_decay=0.0005 --random_splits

# echo "ARMA"
python arma.py --loss=loge --dataset=Cora --num_stacks=2 --num_layers=1 --shared_weights
python arma.py --loss=loge --dataset=Cora --num_stacks=3 --num_layers=1 --shared_weights --random_splits

# echo "APPNP"
python appnp.py --loss=loge --dataset=Cora --alpha=0.1
python appnp.py --loss=loge --dataset=Cora --alpha=0.1 --random_splits

# echo "CiteSeer"
# echo "========"

# echo "GCN"
python gcn.py --loss=loge --dataset=CiteSeer
python gcn.py --loss=loge --dataset=CiteSeer --random_splits

# echo "GAT"
python gat.py --loss=loge --dataset=CiteSeer
python gat.py --loss=loge --dataset=CiteSeer --random_splits

# echo "Cheby"
python cheb.py --loss=loge --dataset=CiteSeer --num_hops=2
python cheb.py --loss=loge --dataset=CiteSeer --num_hops=3 --random_splits

# echo "SGC"
python sgc.py --loss=loge --dataset=CiteSeer --K=2 --weight_decay=0.005
python sgc.py --loss=loge --dataset=CiteSeer --K=2 --weight_decay=0.005 --random_splits

# echo "ARMA"
python arma.py --loss=loge --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights
python arma.py --loss=loge --dataset=CiteSeer --num_stacks=3 --num_layers=1 --shared_weights --random_splits

# echo "APPNP"
python appnp.py --loss=loge --dataset=CiteSeer --alpha=0.1
python appnp.py --loss=loge --dataset=CiteSeer --alpha=0.1 --random_splits

# echo "PubMed"
# echo "======"

# echo "GCN"
python gcn.py --loss=loge --dataset=PubMed
python gcn.py --loss=loge --dataset=PubMed --random_splits

# echo "GAT"
python gat.py --loss=loge --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8
python gat.py --loss=loge --dataset=PubMed --lr=0.01 --weight_decay=0.001 --output_heads=8 --random_splits

# echo "Cheby"
python cheb.py --loss=loge --dataset=PubMed --num_hops=2
python cheb.py --loss=loge --dataset=PubMed --num_hops=2 --random_splits

# echo "SGC"
python sgc.py --loss=loge --dataset=PubMed --K=2 --weight_decay=0.0005
python sgc.py --loss=loge --dataset=PubMed --K=2 --weight_decay=0.0005 --random_splits

# echo "ARMA"
python arma.py --loss=loge --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0
python arma.py --loss=loge --dataset=PubMed --num_stacks=2 --num_layers=1 --skip_dropout=0.5 --random_splits

# echo "APPNP"
python appnp.py --loss=loge --dataset=PubMed --alpha=0.1
python appnp.py --loss=loge --dataset=PubMed --alpha=0.1 --random_splits

# Arxiv

python gcn.py --dataset=Arxiv --runs=20 --batch_norm --no_normalize_features --loss=loge

# echo "GAT"
python gat.py --dataset=Arxiv --runs=20 --batch_norm --no_normalize_features --loss=loge #  --lr=0.01 --weight_decay=0.001 --output_heads=8

# echo "Cheby"
python cheb.py --dataset=Arxiv --runs=20 --no_normalize_features --loss=loge --num_hops=2

# echo "SGC"
python sgc.py --dataset=Arxiv --runs=20 --no_normalize_features --loss=loge --K=2 --weight_decay=0.0005

# echo "ARMA"
python arma.py --dataset=Arxiv --runs=20 --no_normalize_features --loss=loge --num_stacks=2 --num_layers=1 --skip_dropout=0

# echo "APPNP"
python appnp.py --dataset=Arxiv --runs=20 --no_normalize_features --loss=loge --alpha=0.1
