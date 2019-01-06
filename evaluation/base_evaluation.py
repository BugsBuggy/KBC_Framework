from evaluation.new_evaluation import *
from evaluation.classic_evaluation import *
import time
class BaseEvaluation:

    @staticmethod
    def createEvaluation(model, dataset):
        evaluator = None
        try:
            evaluator = globals()[Config.evaluation]()
            evaluator = evaluator.fromConfig(model, dataset)
        except NameError as n:
            print("Evaluation is not defined!")
        return evaluator

# puts embeddings to eval device, evaluates, puts embeddings back to training device
def evaluate_model(model, dataset, epoch, logger):
    model.weights_to_device(Config.eval_device)    # put lookup tables on eval device
    eval = BaseEvaluation.createEvaluation(model, dataset)
    time1 = time.time()
    metric1, metric2 = eval.evaluate(epoch, logger)
    time2 = time.time()
    print("Evaluation Runtime: ", time2-time1)
    model.weights_to_device(Config.device)     # put lookup tables back on training device to continue training
    return metric1, metric2



