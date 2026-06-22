import numpy as np
import torch
from torch.utils.data import Sampler


class BatchTaskSampler(Sampler):
    """Sample task batch"""

    def __init__(
        self,
        labels,
        metatrain_iterations,
        n_way,
        k_shot,
        k_query,
        meta_batch_size=1,
        shuffle=False,
        seed=None,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.seed = seed
        self.shuffle = shuffle

        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

        self.meta_batch_size = meta_batch_size
        self.metatrain_iterations = metatrain_iterations

        # Get label_indeces (e.g. [[0,1],[2,3],...] for labels [n1,n1,n2,n2,...])
        labels = np.array(labels)
        self.label_indeces = [
            torch.from_numpy(np.argwhere(labels == unique_label).reshape(-1))
            for unique_label in np.sort(np.unique(labels))
        ]

    def __len__(self):
        """The number of batch"""
        return self.metatrain_iterations

    def __get_n_ways(self):
        classes = torch.randperm(len(self.label_indeces), generator=self.generator)[: self.n_way]
        return classes

    def __get_k_shots_lists(self):
        support_set = []
        query_set = []

        n_ways = self.__get_n_ways()
        for _, class_ in enumerate(n_ways):
            data_id_list = self.label_indeces[class_.item()]
            data_pos_shuffle = torch.randperm(data_id_list.size()[0], generator=self.generator)

            support_set.append(data_id_list[data_pos_shuffle[: self.k_shot]])
            query_set.append(
                data_id_list[data_pos_shuffle[self.k_shot : self.k_shot + self.k_query]]
            )

        return support_set, query_set

    def __iter__(self):
        """Yeild batch of task with max_batch_size = meta_batch_size"""
        task_batch = []

        for meta_batch_id in range(self.metatrain_iterations * self.meta_batch_size):
            support_set, query_set = self.__get_k_shots_lists()
            task_batch.append([torch.cat(support_set), torch.cat(query_set)])
            if len(task_batch) == self.meta_batch_size:
                res_task_batch = task_batch
                task_batch = []
                yield res_task_batch
