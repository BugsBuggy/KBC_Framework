import torch.nn as nn
from trainer import *
from evaluation.base_evaluation import *
from models.base_model import BaseModel
from evaluation.classic_evaluation import *
from config import *
from util.logger import Logger
import dataset
import time

# create pipeline
if __name__ == '__main__':
    parse()
    confPrint()
    dataset = dataset.load()

    # evaluate string variables before usage
    Config.num_threads = eval(Config.num_threads)
    Config.dimensions = eval(Config.dimensions)
    Config.num_negatives = eval(Config.num_negatives)
    Config.l2_reg = eval(Config.l2_reg)
    Config.num_epochs = eval(Config.num_epochs)
    Config.eval_freq = eval(Config.eval_freq)
    Config.early_stopping = eval(Config.early_stopping)
    Config.patience = eval(Config.patience)
    Config.batch_size = eval(Config.batch_size)
    Config.eval_test_data = eval(Config.eval_test_data)
    Config.topk = eval(Config.topk)
    Config.lifted_reg =eval(Config.lifted_reg)
    Config.lifted_delta = eval(Config.lifted_delta)
    Config.most_frequent_rels = eval(Config.most_frequent_rels)
    Config.train = eval(Config.train)
    Config.evaluate = eval(Config.evaluate)


    stop = False    # is set to true when scores actually do no improve or get worse
    logger = Logger.createLogger()

    if "cuda" in Config.device:
        if not torch.cuda.is_available():
            print("GPU is not available, setting device to CPU...")
            Config.device = "cpu"
            Config.eval_device = "cpu"

    model = None
    if Config.train:
        trainer = Trainer.createTrainer(dataset=dataset)
        stop = trainer.train(Config.num_epochs, Config.eval_freq, logger)
        model = trainer.model


    else:
        # load embeddings and put them to eval device
        model = BaseModel.createModel(dataset.num_entities, dataset.num_relations)
        entity_weights, relation_weights = load_model_embeddings(Config.export_dir, "", Config.emb_file_postfix)
        entity_weights = torch.nn.Parameter(torch.from_numpy(entity_weights))
        relation_weights = torch.nn.Parameter(torch.from_numpy(relation_weights))
        model.entity_emb = nn.Embedding(model.num_entities, model.entity_emb_size, sparse=True, _weight=entity_weights)
        model.relation_emb = nn.Embedding(model.num_relations, model.entity_emb_size, sparse=True,
                                                           _weight=relation_weights)
        model.entity_weights = model.entity_emb.weight
        model.relations_weights = model.relation_emb.weight
        model.weights_to_device(Config.eval_device)

         
    if Config.evaluate and not stop:
        model.weights_to_device(Config.eval_device)
        eval = BaseEvaluation.createEvaluation(model, dataset)
        
        #if not Config.train:
        #    logger = None
        time1 = time.time()
        eval.evaluate(Config.num_epochs, logger)
        time2 = time.time()
        print("Evaluation runtime: ", time2-time1)
