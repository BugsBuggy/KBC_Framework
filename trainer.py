import torch.utils.data
from sampler import *
from models.base_model import *
from evaluation.base_evaluation import *
import itertools
import time

class _Collate:
    def __init__(self, sampler):
        self.sampler = sampler

    def collate(self, batch):
        negs = self.sampler.sample(batch)
        all = itertools.chain(batch,negs)
        return torch.from_numpy(np.fromiter(itertools.chain.from_iterable(all),
            dtype=np.int64, count=3*(len(batch)+len(negs))).reshape((-1,3)))


class Trainer:
    dataset = None
    data = None
    model = None
    sampler = None
    criterion = None
    optimizer = None
    l2_reg = None
    lifted_reg = None
    delta = None
    device = None
    use_early_stop = None
    patience = None
    export_dir = None


    def init(self, data):
        self.model = self.model.to(self.device)
        collate_fn = _Collate(self.sampler)

        self.train_loader = torch.utils.data.DataLoader(
            data,
            Config.batch_size, shuffle=Config.shuffle,
            pin_memory=Config.pin_memory, num_workers=Config.loader_num_workers,
            collate_fn=collate_fn.collate)


    def train(self, num_epochs, eval_freq, logger):
        print("Training...")
        metrics = []    # store metrics -> no improvement or worse metrics after evaluation: stop training
        stagnation = 0  # count the number of times the scores did not improve
        for epoch in range(num_epochs):
            sum_loss = 0
            epoch_time = -time.time()
            forward_time = 0
            backward_time = 0
            optimizer_time = 0
            n_scores = self.sampler.num_negatives + 1
            # put batch into model and train it
            labels = torch.zeros([1]).to(self.device)
            for i, batch in enumerate(self.train_loader):
                n = batch.shape[0]
                n_pos = int(n / n_scores)
                batch = batch.to(self.device)
                if labels.shape != torch.Size([n]):
                    labels = torch.zeros([n], dtype=torch.float).to(self.device)
                    labels[0:n_pos] = 1
                forward_time -= time.time()


                scores = self.model(batch)
                if hasattr(self.criterion, 'margin'):   # margin based loss
                    x1, x2, labels = scores[0], scores[1], scores[2].to(self.device)
                    loss = self.criterion(x1, x2, labels)
                else:
                    loss = self.criterion(scores, labels) * (1 / n)
                    loss += self.model.l2_regularizer(batch, self.l2_reg) * (1 / n)
                    loss += self.model.lifted_constraints(batch, self.lifted_reg, self.delta) * (1 / n)


                sum_loss += loss.item()
                forward_time += time.time()
                # backward pass and optimize
                backward_time -= time.time()
                self.optimizer.zero_grad()
                loss.backward()
                backward_time += time.time()
                optimizer_time -= time.time()
                self.optimizer.step()
                optimizer_time += time.time()
            epoch_time += time.time()

            print("epoch={} progress={:.2f}% loss_item={:.6f} avg_loss={:.6f} forward={:.3f}s backward={:.3f}s opt={:.3f}s other={:.3f}s total={:.3f}s".format(epoch, epoch/num_epochs * 100, loss.item(),  sum_loss / self.train_loader.batch_size, forward_time, backward_time, optimizer_time, epoch_time - forward_time - backward_time - optimizer_time, epoch_time))

            stop, metrics, stagnation_post = self.earlyStopping(epoch, num_epochs, eval_freq, metrics, logger, stagnation, self.patience)
            stagnation = stagnation_post
            if stop:
                return stop

        #self.model.dump_embeddings(self.export_dir)
        return False


    def earlyStopping(self, epoch, num_epochs, eval_freq, metrics, logger, stagnation, patience):
        if epoch and not epoch % eval_freq and not epoch == num_epochs - 1:
            # save metrics and if metrics are worse -> stop training (overfitting)
            metric1, metric2 = evaluate_model(self.model, self.dataset, epoch, logger)
            if self.use_early_stop and len(metrics):
                prev_metric1 = metrics[len(metrics) - 1][0]
                prev_metric2 = metrics[len(metrics) - 1][1]
                if metric1 <= prev_metric1 and metric2 <= prev_metric2:
                    stagnation += 1
                    if stagnation == patience + 1:
                        print("Stopping earlier... metric1: {} <= {} and metric2: {} <= {}".format(metric1, prev_metric1,
                                                                                               metric2, prev_metric2))
                        return True, metrics, stagnation
                else:
                    stagnation = 0
            metrics.append((metric1, metric2))
        return False, metrics, stagnation



    @staticmethod
    def createTrainer(dataset):
        trainer = Trainer()
        if dataset is None:
            trainer.dataset = dataset.load()
        else:
            trainer.dataset = dataset

        trainer.device = torch.device(Config.device)

        trainer.l2_reg = Config.l2_reg
        trainer.lifted_reg = Config.lifted_reg
        trainer.delta = Config.lifted_delta

        torch.set_num_threads(Config.num_threads)

        # use new factory method for model creation
        trainer.model = BaseModel.createModel(trainer.dataset.num_entities, trainer.dataset.num_relations)

        eval(Config.init.format(embs='trainer.model.entity_emb.weight'))
        eval(Config.init.format(embs='trainer.model.relation_emb.weight'))

        trainer.sampler = eval(Config.sampler.format(dataset='trainer.dataset'))
        trainer.criterion = eval(Config.criterion)

        trainer.use_early_stop = Config.early_stopping
        trainer.patience = Config.patience

        trainer.init(trainer.dataset.splits[Config.train_split])
        trainer.export_dir = Config.export_dir
        
        # important: initialize optimizer after putting trainer to gpu (after calling init)
        trainer.optimizer = eval(Config.optimizer.format(parameters='trainer.model.parameters()'))
        return trainer
