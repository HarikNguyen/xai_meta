import unittest
import torch
import numpy as np
from PIL import Image
from loaders.transforms import ToTensor, Normalize, Compose

class TestTransforms(unittest.TestCase):
    def setUp(self):
        """Initialize dummy data for testing"""
        # Create a dummy RGB image (H=84, W=84, C=3) with values [0, 255]
        self.dummy_img = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)

    def test_to_tensor_conversion(self):
        """Verify ToTensor scales data to [0, 1] and permutes to (C, H, W)"""
        transform = ToTensor()
        tensor = transform(self.dummy_img)
        
        # Check shape: (3, 84, 84)
        self.assertEqual(tensor.shape, (3, 84, 84))
        # Check range: must be between 0.0 and 1.0
        self.assertTrue(tensor.max() <= 1.0 and tensor.min() >= 0.0)
        # Check dtype: must be float32
        self.assertEqual(tensor.dtype, torch.float32)

    def test_normalize_values(self):
        """Verify Normalize correctly applies: out = (input - mean) / std"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Normalize(mean, std)
        
        # Create a tensor of 0.5s
        dummy_tensor = torch.full((3, 84, 84), 0.5)
        out = transform(dummy_tensor)
        
        # Manually calculate expected value for the first channel
        expected = (0.5 - mean[0]) / std[0]
        self.assertAlmostEqual(out[0, 0, 0].item(), expected, places=5)
