import unittest
import torch
from loaders.samplers import BatchTaskSampler
from loaders.datasets import MiniImagenetDataset
from loaders.transforms import make_transform

class TestBatchTaskSampler(unittest.TestCase):
    def setUp(self):
        data_root="miniImagenet",
        dataset="miniImagenet",
        dataset_type="train",
        dataset = MiniImagenetDataset(
            data_root=data_root,
            dataset=dataset,
            dataset_type=dataset_type,
            out_path=False,
            transform=make_transform(),
        )
        self.labels = dataset.classes

    def test_sampler_logic(self):
        print("\n--- Running Sampler Logic Test ---")
        n_way, k_shot, k_query = 3, 2, 2
        sampler = BatchTaskSampler(self.labels, 1, n_way, k_shot, k_query, meta_batch_size=1)
        
        # Get the first task
        task = next(iter(sampler))[0]
        support_idx, query_idx = task[0], task[1]
        
        print(f"Sampling: {n_way}-way {k_shot}-shot")
        print(f"Support Indices: {support_idx.tolist()}")
        print(f"Query Indices:   {query_idx.tolist()}")
        
        # Validation
        overlap = set(support_idx.tolist()).intersection(set(query_idx.tolist()))
        print(f"Overlap count: {len(overlap)}")
        
        self.assertEqual(len(overlap), 0, "Support and Query must be disjoint!")
        print("Result: PASS - No data leakage detected.")

if __name__ == '__main__':
    unittest.main()
