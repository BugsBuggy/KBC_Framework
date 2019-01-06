import torch.utils.data
import torch
from util.helpers import *
from config import *
import math
from collections import defaultdict as ddict

class _Collate:
    def __init__(self, ):
        pass

    def collate(self, batch):
        return torch.squeeze(torch.from_numpy(np.array(batch)))


class ER:
    dataset = None
    eval_data = None
    model = None
    device = None
    total_data = None
    topk = None
    most_frequent_rels = None

    def init(self, data):
        self.model = self.model.to(self.device)
        collate_fn = _Collate()
        self.eval_loader = torch.utils.data.DataLoader(
            data,
            Config.eval_batch_size, shuffle=False,
            pin_memory=Config.pin_memory, num_workers=Config.loader_num_workers,
            collate_fn=collate_fn.collate)


    def evaluate(self, epoch, logger):

        raw_ranks_e1 = ddict(list)
        raw_ranks_e2 = ddict(list)
        filtered_ranks_e1_torch = ddict(list)
        filtered_ranks_e2_torch = ddict(list)

        # prepare lookup
        all_triples_e1 = ddict(lambda: ddict(list))          # all entity1's/subjects for a given relation and e2/objects
        all_triples_e2 = ddict(lambda: ddict(list))          # all entity2's/objects for a given relation and e1/subjects

        for e1, r, e2 in self.total_data:
            all_triples_e1[r][e2].append(e1)
            all_triples_e2[r][e1].append(e2)

        all_triples_e1 = dict(all_triples_e1)
        all_triples_e2 = dict(all_triples_e2)


        eval_dict = ddict(list)
        for e1, r, e2 in self.eval_data:
            eval_dict[r].append([e1, r, e2])

        eval_data_prepared = [triple_list for r, triple_list in eval_dict.items() ]

        # evaluate only on the most frequent relations
        if self.most_frequent_rels > 0:
            print("Evaluating on {} most frequent relations...".format(self.most_frequent_rels))
            eval_data_prepared = eval_data_prepared[:self.most_frequent_rels]

        # DataLoader input:
        self.init(eval_data_prepared)
        for i, batch in enumerate(self.eval_loader):

            batch = batch.to(self.device)

            r, E1, E2, multiple_triples = None, None, None, None

            if len(batch.shape) >= 2:
                r_tensor = batch[0][1]
                r = batch[0][1].item()
                E1 = batch[:, 0]
                E2 = batch[:, 2]
                multiple_triples = True

            else:
                # only one test triple for a given relation
                r_tensor = batch[1]
                r = batch[1].item()
                E1 = batch[0]
                E2 = batch[2]

            print("Relation  ", r, "  i: ", i, "  ", round((i/len(self.eval_loader)) * 100, 2), "%")

            self.model.precompute(r_tensor)

            # ---- Swap object/entity 2 ----
            scores_e2 = self.model.scores_e2(E1)
            sorted_e2, sortidx_e2 = torch.sort(scores_e2, descending=True)

            if multiple_triples:
                positions_e1 = [torch.nonzero(sortidx_e2[idx] == torch.unsqueeze(e2, 0)) for idx, e2 in enumerate(E2)]
            else:
                positions_e1 = [torch.nonzero(sortidx_e2 == torch.unsqueeze(E2, 0))]
            ranks_e1_t = [t+1 for t in positions_e1]

            if len(ranks_e1_t) == 1:
                ranks_e1_t = [torch.squeeze(ranks_e1_t[0])]
            ranks_e1_t = torch.LongTensor(ranks_e1_t)

            raw_ranks_e1[r].append(ranks_e1_t)

            E2_np = E2.data.cpu().numpy()
            E1_np = E1.data.cpu().numpy()
            # ---- /Swap object/entity 2 ----

            # ----- FILTERING FOR entity 2 ------
            if multiple_triples:
                remove_idx = [all_triples_e2[r][e].copy() for idx, e in enumerate(E1_np)]

                # prevent test triples from being filtered
                for idx, e in enumerate(remove_idx):
                    e.remove(E2_np[idx])

                for idx, score in enumerate(scores_e2):
                    scores_e2[idx][remove_idx[idx]] = -math.inf

                scores_e2_f, sortidx_e2_f = torch.sort(scores_e2, descending=True)
                positions_e1_f = [torch.nonzero(sortidx_e2_f[idx] == torch.unsqueeze(e2, 0)) for idx, e2 in enumerate(E2)]
            else:
                # if no values to filter -> no need to sort
                positions_e1_f = [torch.nonzero(sortidx_e2 == torch.unsqueeze(E2, 0))]

            f_ranks_e1 = [t + 1 for t in positions_e1_f]
            if len(f_ranks_e1) == 1:
                f_ranks_e1 = [torch.squeeze(f_ranks_e1[0])]
            f_ranks_e1 = torch.LongTensor(f_ranks_e1)

            filtered_ranks_e1_torch[r].append(f_ranks_e1)
            # ----/FILTERING FOR entity 2 ----


            # ---- Swap subject/entity 1 ----
            scores_e1 = self.model.scores_e1(E2)
            sorted_e1, sortidx_e1 = torch.sort(scores_e1, descending=True)

            if multiple_triples:
                positions_e2 = [torch.nonzero(sortidx_e1[idx] == torch.unsqueeze(e1, 0)) for idx, e1 in enumerate(E1)]
            else:
                positions_e2 = [torch.nonzero(sortidx_e1 == torch.unsqueeze(E1, 0))]
            ranks_e2_t = [t + 1 for t in positions_e2]
            if len(ranks_e2_t) == 1:
                ranks_e2_t = [torch.squeeze(ranks_e2_t[0])]
            ranks_e2_t = torch.LongTensor(ranks_e2_t)

            raw_ranks_e2[r].append(ranks_e2_t)
            # ---- /Swap subject/entity 1 ----


            # ----- FILTERING FOR entity 1 ------
            if multiple_triples:
                remove_idx = [all_triples_e1[r][e].copy() for idx, e in enumerate(E2_np)]

                # prevent test triples from being filtered
                for idx, e in enumerate(remove_idx):
                    e.remove(E1_np[idx])

                for idx, score in enumerate(scores_e1):
                    scores_e1[idx][remove_idx[idx]] = -math.inf

                scores_e1_f, sortidx_e1_f = torch.sort(scores_e1, descending=True)

                positions_e2_f = [torch.nonzero(sortidx_e1_f[idx] == torch.unsqueeze(e1, 0)) for idx, e1 in enumerate(E1)]
            else:
                # if no values to filter -> no need to sort
                positions_e2_f = [torch.nonzero(sortidx_e1 == torch.unsqueeze(E1, 0))]

            f_ranks_e2 = [t + 1 for t in positions_e2_f]

            if len(f_ranks_e2) == 1:
                f_ranks_e2 = [torch.squeeze(f_ranks_e2[0])]
            f_ranks_e2 = torch.LongTensor(f_ranks_e2)

            filtered_ranks_e2_torch[r].append(f_ranks_e2)
            # ----/FILTERING FOR entity 1 ----



        e1_rank_list = []
        dummy = [e1_rank_list.append(ranks[0].data.numpy()) for ranks in raw_ranks_e1.values()]
        e2_rank_list = []
        dummy = [e2_rank_list.append(ranks[0].data.numpy()) for ranks in raw_ranks_e2.values()]

        # For PyTorch Version of filtering
        f_e1_ranks_list = []
        dummy = [f_e1_ranks_list.append(ranks[0].data.numpy()) for ranks in filtered_ranks_e1_torch.values()]
        f_e2_ranks_list = []
        dummy = [f_e2_ranks_list.append(ranks[0].data.numpy()) for ranks in filtered_ranks_e2_torch.values()]

        filtered_mrr, filtered_hits_at_ten = self.compute_metrics(e1_rank_list, e2_rank_list, f_e1_ranks_list, f_e2_ranks_list)

        # Save results and store best model
        if logger is not None:
            filtered_mrr = np.round(filtered_mrr, 4)
            filtered_hits_at_ten = np.round(filtered_hits_at_ten, 4)
            logger.log_result(filtered_mrr , filtered_hits_at_ten, epoch, "a")
            logger.compare_best(filtered_mrr, filtered_hits_at_ten, epoch, "_best", self.model)

        return filtered_mrr, filtered_hits_at_ten




    def compute_metrics(self, raw_ranks_e1, raw_ranks_e2, filtered_ranks_e1, filtered_ranks_e2):
        # input -> lists of np arrays (or tensors) -> output one np array
        raw_e1 = np.concatenate((raw_ranks_e1))
        raw_e2 = np.concatenate((raw_ranks_e2))
        raw_ranks = np.concatenate( (raw_e1, raw_e2) )

        filtered_e1 = np.concatenate((filtered_ranks_e1))
        filtered_e2 = np.concatenate((filtered_ranks_e2))
        filtered_ranks = np.concatenate( (filtered_e1, filtered_e2) )

        raw_mrr = np.mean(1.0/ raw_ranks)
        raw_hits_at_ten = np.mean(raw_ranks <= 10).sum() * 100
        raw_hits_at_one = np.mean(raw_ranks <= 1).sum() * 100

        filtered_mrr = np.mean(1.0 / filtered_ranks)
        filtered_hits_at_ten = np.mean(filtered_ranks <= 10).sum() * 100
        filtered_hits_at_one = np.mean(filtered_ranks <= 1).sum() * 100

        print("raw_mrr: ", raw_mrr)
        print("raw_hits@10: " , raw_hits_at_ten)
        print("raw_hits@1: ", raw_hits_at_one)
        print("----------------------------------")
        print("filtered_mrr: ", filtered_mrr)
        print("filtered_hits@10: ", filtered_hits_at_ten)
        print("filtered_hits@1: ", filtered_hits_at_one)
        return filtered_mrr, filtered_hits_at_ten



    @staticmethod
    def fromConfig(model, dataset):
        evaluator = ER()
        if dataset is None:
            evaluator.dataset = dataset.load()
        else:
            evaluator.dataset = dataset

        evaluator.topk = Config.topk
        torch.set_num_threads(Config.num_threads)
        evaluator.device = torch.device(Config.eval_device)
        evaluator.model = model
        evaluator.most_frequent_rels = Config.most_frequent_rels

        if Config.eval_test_data:  # use test data for evaluation, use training and validation data for filtering
            evaluator.eval_data = evaluator.dataset.splits[Config.eval_split]
        else:    # use validation data for evaluation, use training data for filtering
            evaluator.eval_data = evaluator.dataset.splits[Config.valid_split]
            # Training + test + valid

        evaluator.total_data = np.concatenate((evaluator.dataset.splits[Config.train_split], evaluator.dataset.splits[Config.valid_split],
         evaluator.dataset.splits[Config.eval_split]))

        return evaluator
