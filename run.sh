#! /bin/bash

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset wikidata --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1


# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset twitter --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset ppi --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset dblp --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset blogcatalog --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset wikidata1k_multiclass --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1 --multiclass --epochs 100

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset wikidata5k_multiclass --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1 --multiclass --epochs 100

# docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:refinedgae python train_w_feat_small.py --dataset wikidata10k_multiclass --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1 --multiclass --epochs 100
