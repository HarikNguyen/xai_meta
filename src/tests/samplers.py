import unittest
import torch
from loaders.samplers import BatchTaskSampler

class TestSamplers(unittest.TestCase):
    def setUp(self):
        # Assuming we have 10 classes with 10 samples each
        self.labels = []
        for i in range(10):
            self.labels.extend([f"class_{i}"] * 10)
            
    def test_sampler_output_structure(self):
        n_way, k_shot, k_query = 5, 1, 1
        meta_batch_size = 2
        iterations = 10
        
        sampler = BatchTaskSampler(
            labels=self.labels,
            metatrain_iterations=iterations,
            n_way=n_way, k_shot=k_shot, k_query=k_query,
            meta_batch_size=meta_batch_size
        )
        
        # Get a batch from the sampler
        batch = next(iter(sampler))
        
        # Check the batch size
        self.assertEqual(len(batch), meta_batch_size)
        
        # Check the structure of the first task in the batch
        task = batch[0]
        self.assertEqual(len(task[0]), n_way * k_shot)
        self.assertEqual(len(task[1]), n_way * k_query)

    def test_no_leakage(self):
        """Check that there is no overlap between support and query sets."""
        sampler = BatchTaskSampler(self.labels, 1, 2, 5, 5, 1)
        batch = next(iter(sampler))
        support_idx, query_idx = batch[0]
        
        overlap = set(support_idx.tolist()).intersection(set(query_idx.tolist()))
        self.assertEqual(len(overlap), 0, "Leakage detected!")
