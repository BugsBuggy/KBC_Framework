import torch.utils.data
import torch
import math
from util.helpers import *
from collections import defaultdict as ddict

class _Collate:
    def __init__(self, ):
        pass

    def collate(self, batch):
        return torch.squeeze(torch.from_numpy(np.array(batch)))


class PR:
    dataset = None
    eval_data = None
    model = None
    device = None
    most_frequent_rels = None

    test_data = None
    train_data = None
    valid_data = None
    eval_test_data = None

    topk = None

    def init(self, data):
        self.model = self.model.to(self.device)
        collate_fn = _Collate()
        self.eval_loader = torch.utils.data.DataLoader(
            data,
            Config.eval_batch_size, shuffle=False,
            pin_memory=Config.pin_memory, num_workers=Config.loader_num_workers,
            collate_fn=collate_fn.collate)

    def count_e1_e2_by_relation(self, data):
        rel_map = ddict(int)
        for r in data.keys():
            rel_map[r] = len(data[r])
        count_pairs_by_relation = rel_map.items()
        count_pairs_by_relation = sorted(count_pairs_by_relation, key=lambda x: -x[1])
        return count_pairs_by_relation

    # computes the position of a tuple for the flattened 1d score matrix
    def convert_idx_to_1d(self, tuples_r, n=None):
        if n is None:
            n = self.model.num_entities
        pos_1d = []
        row_idx, column_idx = tuples_r
        for i in range(len(row_idx)):
            pos_1d.append(row_idx[i] * n + column_idx[i])
        return pos_1d


    def evaluate(self, epoch, logger):
        #prepare data
        idx_train = ddict(list)
        for e1, r, e2 in self.train_data:
            idx_train[r].append((e1, e2))

        if self.eval_test_data:
            idx_valid = ddict(list)
            for e1, r, e2 in self.valid_data:
                idx_valid[r].append((e1, e2))

        idx_test = ddict(list)
        for e1, r, e2 in self.test_data:
            idx_test[r].append((e1, e2))

        tuples_by_relation = self.count_e1_e2_by_relation(idx_test)

        relations = np.array([x[0] for x in tuples_by_relation])
        #tuples_count = np.array([x[1] for x in tuples_by_relation])

        # speedup grid search
        if self.most_frequent_rels > 0:
            print("Evaluating on {} most frequent relations...".format(self.most_frequent_rels))
            relations = relations[:self.most_frequent_rels]

        prepare_test = ddict(list)
        for e1, r, e2 in self.test_data:
            prepare_test[r].append([e1, r, e2])

        # sorted data
        prepare_test_sorted = ddict(list)
        for r in relations:
            prepare_test_sorted[r].append(prepare_test[r])

        eval_data_prepared = [triple_list for r, triple_list in prepare_test_sorted.items()]

        ranks_by_r = ddict(list)
        num_true_triples = ddict(list)


        self.init(eval_data_prepared)
        for i, batch in enumerate(self.eval_loader):

            batch = batch.to(self.device)
            r = None

            if len(batch.shape) >= 2:
                r_tensor = batch[0][1]
                r = batch[0][1].item()

            else:
            # only one test triple for a given relation
                r_tensor = batch[1]
                r = batch[1].item()
            print("Evaluating: {}   Progress: {}%".format(r, round(i/len(self.eval_loader) * 100, 2)))
            scores = ddict(list)

            score_matrix = self.model.score_matrix_r(r_tensor)
            scores[r].append(score_matrix)

            # ----- FILTERING -----
            # all e1, e2 for a given relation in test, validation data
            tuples_r_test = np.array(prepare_test_sorted[r][0])
            tuples_r_test = [tuples_r_test[:,0], tuples_r_test[:,2]]

            tuples_r_train = np.array(idx_train[r])
            tuples_r_train = [tuples_r_train[:,0], tuples_r_train[:,1]]

            score_matrix[tuples_r_train] = -math.inf     # Filter training set out

            # Filter validation set out
            if self.eval_test_data:
                tuples_r_valid = np.array(idx_valid[r])
                if (len(tuples_r_valid) > 0):
                    tuples_r_valid = [tuples_r_valid[:, 0], tuples_r_valid[:, 1]]
                    score_matrix[tuples_r_valid] = -math.inf

            # ---- /FILTERING -----

            test_tuples_r_1d = self.convert_idx_to_1d(tuples_r_test)
            num_true_triples[r] = len(test_tuples_r_1d)
            test_tuples_r_1d_tensor = torch.squeeze(torch.LongTensor([test_tuples_r_1d]))
            topk = self.compute_topk(score_matrix, test_tuples_r_1d_tensor)
            ranks = topk.cpu().data.numpy()
            if len(ranks.shape) > 0:
                ranks = np.sort(ranks)
            print(ranks)
            ranks_by_r[r].append(ranks)

        print("-----------------------")
        avg_map, avg_hits = self.metrics(ranks_by_r, num_true_triples)

        print("TOTAL MAP: {} ".format(avg_map))
        print("TOTAL HITS: {}".format(avg_hits))

        # save results
        if logger is not None:
            avg_map = round(avg_map, 4)
            avg_hits = round(avg_hits, 4)
            logger.log_result(avg_map, avg_hits, epoch, "a")
            logger.compare_best(avg_map, avg_hits, epoch, "_best", self.model)

        return avg_map, avg_hits



    def compute_topk(self, score_matrix, tuples_r_1d):
        score_matrix = score_matrix.reshape((1, -1)).flatten()

        if len(score_matrix) > self.topk+1:
            sorted_k_values, sorted_k_indexs = torch.topk(score_matrix, self.topk, largest=True, sorted=True)

        other = torch.zeros(len(sorted_k_indexs)).long().to(self.device)

        tuples_r_1d = tuples_r_1d.to(self.device)

        if len(tuples_r_1d.size()) > 0:
            check = [torch.where(sorted_k_indexs == t, sorted_k_indexs, other) for t in tuples_r_1d if len(torch.nonzero(sorted_k_indexs == t)) > 0]
        else:
            check = [torch.where(sorted_k_indexs == tuples_r_1d, sorted_k_indexs, other)]

        ranks = [torch.nonzero(t)+1 for t in check]
        if len(ranks) == 1: # one or zero elements in ranks
            ranks = ranks[0] if len(ranks[0].size()) <= 1 else ranks[0][0]
        else:
            ranks = torch.LongTensor(ranks).to(self.device)

        return ranks


    def metrics(self, ranks_by_relation, num_true_triples):
        total_precision = 0
        normalization = 0
        total_hits = 0
        for r, ranks in ranks_by_relation.items():
            total_hits += len(ranks[0])
            normalization += min(num_true_triples[r], self.topk)
            for idx, rank in enumerate(ranks[0]):
                total_precision += (idx + 1) / rank

        avg_map = (total_precision / normalization) * 100
        avg_hits = (total_hits / normalization) * 100
        return avg_map, avg_hits


    @staticmethod
    def fromConfig(model, dataset):
        evaluator = PR()
        if dataset is None:
            evaluator.dataset = dataset.load()
        else:
            evaluator.dataset = dataset

        evaluator.device = torch.device(Config.eval_device)

        torch.set_num_threads(Config.num_threads)
        evaluator.model = model

        coder = Coder()
        data_dir = Config.data_dir
        dataset = Config.dataset
        train_triples = read_triplets(data_dir + Config.dataset + "/" + Config.raw_split_files['train'], None)
        train_triples = coder.construct_encoder(train_triples)

        test_triples = read_triplets(data_dir + dataset + "/" + Config.raw_split_files['test'], coder)
        test_triples = coder.construct_encoder(test_triples)

        valid_triples = read_triplets(data_dir + dataset + "/" + Config.raw_split_files['valid'], coder)
        valid_triples = coder.construct_encoder(valid_triples)


        evaluator.train_data = train_triples
        evaluator.eval_test_data = Config.eval_test_data

        if Config.eval_test_data:    # use test set for evaluation, training and validation split for filtering
            evaluator.test_data = test_triples
            evaluator.valid_data = valid_triples
        else:    # use validation set for evaluation and training set for filtering
            evaluator.test_data = valid_triples

        evaluator.most_frequent_rels = Config.most_frequent_rels
        evaluator.topk = Config.topk

        return evaluator




 
