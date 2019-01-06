import numpy as np
import dataset


class Sampler:
    def __init__(self, num_negatives):
        self.num_negatives = num_negatives

    # generate negative samples for the given batch (list of tensors)
    # input is: [pos0,pos1,pos2,...]
    # output is: [neg0,neg1,neg2,...,neg0,neg1,neg2,...]
    def sample(self, batch, out=None):
        pass


class PerturbOneSampler(Sampler):
    """ Classic sampler: replaces either subject or object with random entity """
    def __init__(self, dataset, num_negatives):
        super(PerturbOneSampler, self).__init__(num_negatives)
        self.dataset = dataset

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out)!=n_neg:
            out = [batch[i%n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i%n_pos]

        # perturb
        which = np.random.randint(0,2,n_neg)*2
        what = np.random.randint(0,self.dataset.num_entities,n_neg)
        for i in range(n_neg):
           out[i][which[i]] = what[i]

        return out


class PerturbOneSampler_PC(Sampler):
    def __init__(self, dataset, num_negatives):
        super(PerturbOneSampler_PC, self).__init__(num_negatives)
        self.dataset = dataset

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out)!=n_neg:
            out = [batch[i%n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i%n_pos]

        # perturb
        what = np.random.randint(0,self.dataset.num_entities,n_neg)
        for i in range(n_neg):
           out[i][2] = what[i]

        return out


class PerturbOneRelSampler(Sampler):
    """ Replaces either subject, relation or object with random entity/relation """
    def __init__(self, dataset, num_negatives):
        super(PerturbOneRelSampler, self).__init__(num_negatives)
        self.dataset = dataset

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out)!=n_neg:
            out = [batch[i%n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i%n_pos]

        # perturb
        which = np.random.randint(0,3,n_neg)
        what = np.random.randint(0,self.dataset.num_entities,n_neg)
        rels = np.random.randint(0,self.dataset.num_relations,n_neg)
        for i in range(n_neg):
            if which[i] == 1:
                out[i][which[i]] = rels[i]
            else:
                out[i][which[i]] = what[i]

        return out


class PerturbTwoSampler(Sampler):
    """ Replaces both subject and object, each with a random entity """
    def __init__(self, dataset, num_negatives):
        super(PerturbTwoSampler, self).__init__(num_negatives)
        self.dataset = dataset

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out)!=n_neg:
            out = [batch[i%n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i%n_pos]

        # perturb
        sub = np.random.randint(0,self.dataset.num_entities,n_neg)
        obj = np.random.randint(0,self.dataset.num_entities,n_neg)
        for i in range(n_neg):
            out[i][0] = sub[i]
            out[i][2] = obj[i]

        return out


class PerturbTwoRelSampler(Sampler):
    """ Replaces all, subject, relation and object, each with a random entity/relation """
    def __init__(self, dataset, num_negatives):
        super(PerturbTwoRelSampler, self).__init__(num_negatives)
        self.dataset = dataset

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out)!=n_neg:
            out = [batch[i%n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i%n_pos]

        # perturb
        sub = np.random.randint(0,self.dataset.num_entities,n_neg)
        rel = np.random.randint(0,self.dataset.num_relations,n_neg)
        obj = np.random.randint(0,self.dataset.num_entities,n_neg)
        for i in range(n_neg):
            out[i][0] = sub[i]
            out[i][1] = rel[i]
            out[i][2] = obj[i]

        return out


class PerturbOneTypeConSampler(Sampler):
    """ Replaces either subject or object with a random entity which is type consistent with relation's domain/range """
    def __init__(self, dataset, num_negatives):
        super(PerturbOneTypeConSampler, self).__init__(num_negatives)
        self.dataset = dataset
        self.subs, self.objs = dataset.load_subs_objs()

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out) != n_neg:
            out = [batch[i % n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i % n_pos]

        # perturb
        which = np.random.randint(0, 2, n_neg)*2
        for i in range(n_neg):
            if which[i]:
                what = np.random.randint(0, len(self.objs[out[i][1]]))
                out[i][which[i]] = self.objs[out[i][1]][what]
            else:
                what = np.random.randint(0, len(self.subs[out[i][1]]))
                out[i][which[i]] = self.subs[out[i][1]][what]

        return out


class PerturbTwoTypeConSampler(Sampler):
    """ Replaces both subject and object with a random entity which is type consistent with relation's domain/range """
    def __init__(self, dataset, num_negatives):
        super(PerturbTwoTypeConSampler, self).__init__(num_negatives)
        self.dataset = dataset
        self.subs, self.objs = dataset.load_subs_objs()

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out) != n_neg:
            out = [batch[i % n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i % n_pos]

        # perturb
        for i in range(n_neg):
            sub = np.random.randint(0, len(self.subs[out[i][1]]))
            out[i][0] = self.subs[out[i][1]][sub]
            obj = np.random.randint(0, len(self.objs[out[i][1]]))
            out[i][2] = self.objs[out[i][1]][obj]

        return out


class PerturbOneTypeIncSampler(Sampler):
    """ Replaces either subject or object with an entity which is type inconsistent with relation's domain/range """
    def __init__(self, dataset, num_negatives):
        super(PerturbOneTypeIncSampler, self).__init__(num_negatives)
        self.dataset = dataset
        self.subs, self.objs = dataset.load_subs_objs()
        self.not_subs = []
        self.not_objs = []
        all_ents = list(range(dataset.num_entities))
        for i in range(len(self.subs)):
            if len(self.subs[i]) == self.dataset.num_entities:
                self.not_subs.append(all_ents)
            else:
                self.not_subs.append(list(set(all_ents) - set(self.subs[i])))

        for i in range(len(self.objs)):
            if len(self.objs[i]) == self.dataset.num_entities:
                self.not_objs.append(all_ents)
            else:
                self.not_objs.append(list(set(all_ents) - set(self.objs[i])))

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out) != n_neg:
            out = [batch[i % n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i % n_pos]

        # perturb
        which = np.random.randint(0, 2, n_neg)*2
        for i in range(n_neg):
            if which[i]:
                what = np.random.randint(0, len(self.not_objs[out[i][1]]))
                out[i][which[i]] = self.not_objs[out[i][1]][what]
            else:
                what = np.random.randint(0, len(self.not_subs[out[i][1]]))
                out[i][which[i]] = self.not_subs[out[i][1]][what]

        return out


class PerturbTwoTypeIncSampler(Sampler):
    """ Replaces both subject and object with an entity which is type inconsistent with relation's domain/range """
    def __init__(self, dataset, num_negatives):
        super(PerturbTwoTypeIncSampler, self).__init__(num_negatives)
        self.dataset = dataset
        self.subs, self.objs = dataset.load_subs_objs()
        self.not_subs = []
        self.not_objs = []
        all_ents = list(range(dataset.num_entities))
        for i in range(len(self.subs)):
            if len(self.subs[i]) == self.dataset.num_entities:
                self.not_subs.append(all_ents)
            else:
                self.not_subs.append(list(set(all_ents) - set(self.subs[i])))

        for i in range(len(self.objs)):
            if len(self.objs[i]) == self.dataset.num_entities:
                self.not_objs.append(all_ents)
            else:
                self.not_objs.append(list(set(all_ents) - set(self.objs[i])))

    # @profile
    def sample(self, batch, out=None):
        # TODO make efficient
        n_pos = len(batch)  # TODO assumes each tensor has 1 row
        n_neg = n_pos*self.num_negatives

        if out is None or len(out) != n_neg:
            out = [batch[i % n_pos].copy() for i in range(n_neg)]
        else:  # reuse
            for i in range(n_neg):
                out[i][:] = batch[i % n_pos]

        # perturb
        for i in range(n_neg):
            sub = np.random.randint(0, len(self.not_subs[out[i][1]]))
            out[i][0] = self.not_subs[out[i][1]][sub]
            obj = np.random.randint(0, len(self.not_objs[out[i][1]]))
            out[i][2] = self.not_objs[out[i][1]][obj]

        return out
