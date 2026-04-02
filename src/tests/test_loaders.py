import unittest
import torch
from torch.utils.data import DataLoader
from loaders.samplers import BatchTaskSampler
from loaders.datasets import MiniImagenetDataset
from loaders.transforms import make_transform
from loaders import _task_collate 

class TestLoaderMiniImageNet(unittest.TestCase):
    def setUp(self):
        print("\n" + "="*60)
        print("SETTING UP UNIT TEST")
        print("="*60)
        
        self.n_way, self.k_shot, self.k_query = 5, 1, 15
        self.meta_batch_size = 4
        self.metatrain_iterations = 4
        print(f"Preparing with:\n\tn_way: {self.n_way}\n\tk_shot: {self.k_shot}\n\tk_query: {self.k_query}\n\tmeta_batch_size: {self.meta_batch_size}\n\tmetatrain_iterations: {self.metatrain_iterations}") 
        
        self.dataset = MiniImagenetDataset(
            root_dir="miniImagenet",
            dataset_name="miniImagenet",
            dataset_type="train",
            out_path=False,
            transform=make_transform(),
        )
        seed = None
        sample = {
            "metatrain_iterations": self.metatrain_iterations,
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "k_query": self.k_query,
            "meta_batch_size": self.meta_batch_size,
            "shuffle": True,
        }
        self.sampler = BatchTaskSampler(self.dataset.classes, seed=seed, **sample)

        num_workers = 2
        self.loader = DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_task_collate,
        )

    def test_dataset_logic(self):
        print("\n--- Running Dataset Test ---")
        img, label = self.dataset[0]
        print(f"Single Image Shape: {img.shape}")
        print(f"Raw Label (Global ID): {label}")

        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(label, (int, str))

        indices = [0, 1, 2]
        batch_data = self.dataset[indices]
        print(f"Fetched {len(batch_data)} items.")
        self.assertEqual(len(batch_data), len(indices))

        # Check first item in the returned batch
        first_img, first_lbl = batch_data[0]
        self.assertEqual(first_img.shape, img_single.shape)
        print(f"First image shape: {first_img.shape}")

        print("Check passed!")


    def test_sampler_logic(self):
        print("\n--- Running Sampler Logic Test ---")
        
        # Get the first task
        task = next(iter(self.sampler))[0]
        support_idx, query_idx = task[0], task[1]
        n_way, k_shot, k_query = self.n_way, self.k_shot, self.k_query
        print(f"Sampling: {n_way}-way {k_shot}-shot {k_query}-query")
        print(f"Support Indices: {support_idx.tolist()}")
        print(f"Query Indices:   {query_idx.tolist()}")
        
        # Validation
        overlap = set(support_idx.tolist()).intersection(set(query_idx.tolist()))
        print(f"Overlap count: {len(overlap)}")
        
        self.assertEqual(len(overlap), 0, "Support and Query must be disjoint!")
        print("Result: PASS - No data leakage detected.")

    def test_collator_logic(self):
        print("\n--- Running Collator Logic Test ---")


    def test_loader_logic(self):
        print("\n--- Running Loader Logic Test ---")

    def tearDown(self):
        print("\n" + "-"*60)
        print("TEST CASE COMPLETED")
        print("-"*60)

if __name__ == '__main__':
    unittest.main()
