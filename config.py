import argparse
import json
import torch.nn

class Config:
    export_dir = None
    log_file = None  # name of the log file
    num_threads = None

    # Comment
    comment = None

    # Dataset
    dataset = None
    data_dir = 'data/'

    raw_split_files = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
    split_files = {'train': 'train.csv', 'valid': 'valid.csv', 'test': 'test.csv'}
    relation_index_file = "rel_index.del"
    entity_index_file = "entity_index.del"

    train = None
    evaluate = None
    emb_file_postfix = None   #Only for train=False & evaluate=True: which embedding files to load

    # Model
    ent_func = None
    model = None
    dimensions = None
    criterion = 'nn.BCEWithLogitsLoss()'   # 'nn.MarginRankingLoss(margin=2.0)'
    num_negatives = None
    lr = None
    sampler =  None
    optimizer = None
    l2_reg = None
    init = None

    # Training
    num_epochs = None
    early_stopping = None
    patience = None
    eval_freq = None
    device = None
    batch_size = None
    shuffle = True           # Setting this to False may interfere with the Training process
    pin_memory = False       # Setting this to True may interfere with GPU
    loader_num_workers = 0

    train_split = 'train'  # train, valid, test    data used for training
    eval_split = 'test'  # train, valid, test      data used for evaluation
    valid_split = 'valid' # train, valid, test     data used for validation

    # Evaluation     
    evaluation = None
    eval_device = None
    eval_test_data = None
    topk = None
    eval_batch_size = 1    # leave this at 1
    most_frequent_rels = '0'

    # Lifted constraints
    lifted_reg = None
    lifted_delta = None

    # special settings from TranseE
    norm = None
    L1 = None

    # Special settings for ConvE
    drop1 = None
    drop2 = None
    drop_channel = None
    model_path = None


def parse():
    parser = argparse.ArgumentParser()
    for k, v in sorted(Config.__dict__.items(), key=lambda x: x[0]):
        if not k.startswith("__") and not callable(v) and not v == "dataset":
            if v is None:
                parser.add_argument("--" + k, type=str, default=v)
            elif type(v) == list:
                parser.add_argument("--" + k, type=type(v[0]), nargs="+", default=v)
            elif type(v) == dict:
                parser.add_argument("--" + k, type=json.loads, default=v)
            else:
                parser.add_argument("--" + k, type=type(v), default=v)
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    Config.dataset = args.dataset
    for k, v in sorted(Config.__dict__.items(), key=lambda x: x[0]):
        if not k.startswith("__") and not callable(v) and not v == "dataset":
            setattr(Config, k, args.__dict__[k])


def confPrint():
    for k, v in sorted(Config.__dict__.items(), key=lambda x: x[0]):
        if not k.startswith("__") and not callable(v):
            print("{:>25} {}".format("--" + k, repr(v)))


