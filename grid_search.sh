#!/bin/bash
# General

export_dir='emb_folder'
log_file='log_DistMult'    # name of the log file
num_threads='1'

# Comment
comment='~~'

# Dataset
dataset='FB15k'           #'FB15k-237', 'WN18', 'WN18RR'

# Model
ent_func='identity'       # Sigmoid(), ReLU(), identity

model='DistMult'          # DistMult, ConvE, TransE, Analogy, Complex
dimensions='200 152 100'

train='True'
evaluate='True'


num_negatives='6'
sampler='PerturbTwoSampler'   # 'PerturbOneSampler', 'PerturbOneRelSampler'
lr='0.1 0.01'
optimizer='Adagrad'    # SparseAdam, SGD
l2_reg='0.00001 0.000001 0.0000001'

# Initialization
init='uniform_({embs},-0.01,0.01)'  # uniform_({embs},-0.01,0.01), xavier_uniform_({embs},gain=1)

# Training
num_epochs='150'
eval_freq='50'           # Evaluates after eval_freq epochs to test the model performance
early_stopping='False'   # Stop when valid/test metrics do not improve or get worse, only possible when eval_freq < num_epochs
patience='0'
device='cuda:0'          # cpu, gpu
batch_size='100'


# Evaluation
evaluation='PR'          # ER, PR
eval_device='cuda:0'     # Evaluation device
topk='100'               # Only for PR: Compute metrics for Top K triples
eval_test_data='False'   # True, False     True: use test data for evaluation, False: use validation data for evaluation
most_frequent_rels='30'  # Only Evaluate on the most frequent relations to abbreviate grid search

# lifted constraints
lifted_reg='0.0'
lifted_delta='0.0'



# PASS VARIABLES TO CONFIG HERE

counter=1

for dim in $dimensions
do
  for l in $lr
  do
    for l_r in $lifted_reg
    do
      for l_d in $lifted_delta
      do
        for ini in $init
        do
          for l2 in $l2_reg
           do
             for sam in $sampler
             do
                echo "Running combination: $counter"
                python3 main.py --export_dir $export_dir  --log_file $log_file --num_threads $num_threads --comment $comment \
                                --ent_func $ent_func --model $model --dimensions $dim --train $train --evaluate $evaluate --num_negatives $num_negatives \
                                --sampler $sam'({dataset},'$num_negatives')' --lr $l --optimizer 'torch.optim.'$optimizer'({parameters}, lr='$l')' --l2_reg $l2 \
                                --init 'nn.init.'$ini --num_epochs $num_epochs --eval_freq $eval_freq --early_stopping $early_stopping --patience $patience --device $device --batch_size $batch_size \
                                --evaluation $evaluation --eval_device $eval_device --topk $topk --eval_test_data $eval_test_data \
                                --most_frequent_rels $most_frequent_rels --lifted_reg $l_r --lifted_delta $l_d \
                                $dataset
                let "counter++"
            done
          done
        done
      done
    done
  done
done
