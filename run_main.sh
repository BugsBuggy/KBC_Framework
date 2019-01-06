#!/bin/bash

# General
export_dir='emb_folder'                  # directory to export logs and embedding/relation files
log_file='log'                           # name of the log file
num_threads='1'

# Comment
comment='~~'

# Dataset
dataset='WN18'                          # WN18RR, FB15k, FB15k-237

# Model
ent_func='Sigmoid()'                     # Sigmoid(), ReLU(), identity
model='DistMult'                        # DistMult, Analogy, Complex
dimensions='200'

train='True'
evaluate='True'
emb_file_postfix="_best_DistMult_ER"     # Which embedding files to load: R?.dat, E?.dat with ?=emb_file_postfix", only in case of evaluate=True, Train=False,

num_negatives='6'
sampler='PerturbOneRelSampler'           # PerturbOneSampler, PerturbTwoSampler, PerturbOneTypeConSampler, PerturbOneTypeIncSampler
lr='0.1'
l2_reg='0.000001'
optimizer='Adagrad'                      # SparseAdam, SGD

# Initialization
init='uniform_({embs},-0.01,0.01)'       # uniform_({embs},-0.01,0.01), xavier_uniform_({embs},gain=1)

# Training
num_epochs='1'
eval_freq='50'
device='cuda:0'                          # cpu, cuda:x
batch_size='100'

early_stopping='False'                   # Stop when valid/test metrics do not improve or get worse, only possible when eval_freq < num_epochs
patience='1'                             # number of times to observe worsening of metrics until early stopping

# Evaluation
evaluation='ER'                          # ER: Entity ranking, PR: Entity-Pair Ranking
eval_device='cuda:0'                     # cpu, cuda:x
topk='100'                               # Only for PR: Compute metrics for Top K triples
eval_test_data='True'                    # True, False     True: use test data for evaluation, False: use validation data for evaluation
most_frequent_rels='0'                   # Evaluate only on the most frequent relations, 0 to deactivate

# lifted constraints
lifted_reg='0.0'
lifted_delta='0.0'


# PASS VARIABLES TO CONFIG HERE

                    python3 main.py --export_dir $export_dir  --log_file $log_file --num_threads $num_threads --comment $comment \
                    --ent_func $ent_func --model $model --dimensions $dimensions --train $train --evaluate $evaluate --emb_file_postfix $emb_file_postfix --num_negatives $num_negatives \
                    --sampler $sampler'({dataset},'$num_negatives')' --lr $lr --optimizer 'torch.optim.'$optimizer'({parameters}, lr='$lr')' --l2_reg $l2_reg \
                    --init 'nn.init.'$init --num_epochs $num_epochs --eval_freq $eval_freq --early_stopping $early_stopping --patience $patience --device $device --batch_size $batch_size \
                    --evaluation $evaluation --eval_device $eval_device --topk $topk --eval_test_data $eval_test_data \
                    --most_frequent_rels $most_frequent_rels --lifted_reg $lifted_reg --lifted_delta $lifted_delta \
                    $dataset
