#!/bin/bash
#SBATCH --job-name=protopool_aug
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=dgxmatinf


export TRAIN_DIR=/shared/sets/datasets/birds/train_birds_augmented/train_birds_augmented/train_birds_augmented/
export PUSH_DIR=/shared/sets/datasets/birds/train_birds/train_birds/train_birds/
export TEST_DIR=/shared/sets/datasets/birds/test_birds/test_birds/test_birds/

export ARCH=resnet50

export RESULTS=/shared/results/sacha/local_prototypes/proto_pool/inat50_aug

python main.py --data_type birds --num_classes 200 --batch_size 80 --num_descriptive 10 --num_prototypes 202 --results $RESULTS --use_scheduler --arch $ARCH --pretrained --proto_depth 256 --warmup_time 10 --warmup --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --data_train $TRAIN_DIR --data_push $PUSH_DIR --data_test $TEST_DIR --inat --augmentations