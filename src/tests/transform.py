import unittest
import torch
import numpy as np
from src.transforms import ToTensor, Normalize, Compose

class TestTransforms(unittest.TestCase):
    def test_to_tensor(self):
        # Create a dummy image (H, W, C)
        dummy_img = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
        transform = ToTensor()
        tensor = transform(dummy_img)
        
        self.assertEqual(tensor.shape, (3, 84, 84))
        self.assertTrue(tensor.max() <= 1.0)
        self.assertTrue(tensor.min() >= 0.0)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_normalize(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Normalize(mean, std)
        dummy_tensor = torch.ones((3, 84, 84)) * 0.5
        out = transform(dummy_tensor)
        
        # Check that the output tensor has the expected mean and standard deviation
        expected = (0.5 - mean[0]) / std[0]
        self.assertAlmostEqual(out[0, 0, 0].item(), expected, places=5)
