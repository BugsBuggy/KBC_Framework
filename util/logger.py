from config import Config
import csv
import os

class Logger():
    metric = None
    path = "logs/"

    @staticmethod
    def createLogger():
        logger = Logger()
        if Config.evaluation == "PR":    # or create separate logging methods for ER and PR, but this is shorter
            logger.metric = "MAP"
        elif Config.evaluation == "ER":
            logger.metric = "MRR"
        else:
            logger.metric = "Undefined"
            print("Evaluation protocol is Undefined!")

        if not os.path.exists(logger.path):
            os.makedirs(logger.path)

        return logger


    def log_result(self, map_mrr, hits, epoch, mode, postfix=""):
        file_name = self.path + Config.log_file + postfix + '_' + Config.evaluation + '.csv'
        
        s = ""
        if not os.path.exists(file_name) or mode == "r+":
            with open(file_name, 'w') as f_new:
                head = (";".join(["dataset", "model", self.metric, "Hits", "comment",
                                      "opt", "sampler", "init", "ent_func", "epoch",
                                      "dim", "lr", "l2_reg", "test_data", "topk", "lifted_reg", "lifted_delta", "early_stopping", "patience"]))
                if mode == "a":
                    f_new.write(head + "\n")
                else:
                    s += head + "\n"
                f_new.write(s)
                print("Created", file_name)

        with open(file_name, mode) as f:
            s += ";".join(
                [Config.dataset, Config.model, str(map_mrr), str(hits), Config.comment, Config.optimizer, Config.sampler,
                 Config.init,
                 Config.ent_func, str(epoch), str(Config.dimensions), str(Config.lr), str(Config.l2_reg),
                 str(Config.eval_test_data),
                 str(Config.topk), str(Config.lifted_reg), str(Config.lifted_delta), str(Config.early_stopping), str(Config.patience)])
            s += "\n"
            f.write(s)

    # compare if metric 1,2 are better than best stored metrics,
    # overwrite if it is the case
    def compare_best(self, metric1, metric2, epoch, postfix, model):
        file_name = self.path + Config.log_file + postfix + "_" + Config.model + "_" + Config.evaluation + ".csv"

        if not os.path.exists(file_name):
            with open(file_name, 'w'): pass

        with open(file_name, "r+") as f:
            reader = csv.reader(f, delimiter=';')
            lines = list(reader)
            postfix = "_best_" + Config.model
            if not len(lines) <= 1:
                if float(lines[1][2]) < metric1 and float(lines[1][3]) < metric2:
                    self.log_result(metric1, metric2, epoch, "r+", postfix)
                    model.dump_embeddings(Config.export_dir, postfix + "_"  + Config.evaluation)
            else: # first time logging
                self.log_result(metric1, metric2, epoch, "r+", postfix)
                model.dump_embeddings(Config.export_dir, postfix + "_" + Config.evaluation)