import os
import numpy as np
from PIL import Image
from typing import List


class MiniImagenetDataset(object):

    def __init__(
        self,
        root_dir,
        dataset_name,
        dataset_type,
        transform=None,
        out_path=False,
    ):
        # Get data points
        self.root_dir = root_dir
        self.dataset_name = dataset_name

        assert dataset_type in ["train", "val", "test"]
        self.dataset_type = dataset_type
        self.transform = transform
        self.out_path = out_path
        filenames, classes = self.__get_filenames(dataset_type)
        self.filenames = filenames
        self.classes = classes

    def __get_filenames(self, dataset_type):
        classnames = os.listdir(os.path.join(self.root_dir, dataset_type))
        filenames = []
        classes = []
        for classname in classnames:
            for filename in os.listdir(
                os.path.join(self.root_dir, dataset_type, classname)
            ):
                filenames.append(
                    os.path.join(self.root_dir, dataset_type, classname, filename)
                )
                classes.append(classname)
        return filenames, classes

    def __len__(self):
        """Length of dataset"""
        return len(self.filenames)

    def __getitem__(self, index):
        """Load image and its label by index

        Args
        ----------
        index: int
            Index of datapoint. index must be less than dataset length

        Returns
        ----------
        (image, label, img_path) if out_path is True
        (image, label) if out_path is False
        """

        label_name = self.filenames[index].split("/")[-2]
        img_path = self.filenames[index]

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            img = self.transform(img)
        # return
        if self.out_path:
            return img, label_name, img_path
        return img, label_name

    def __getitems__(self, indices: List):
        """Load images and their label for support and query sets

        Args
        -----------
        indices: List
            support indices and query indices from sample_iter

        Returns
        -----------
        List[support_set, query_set]
        """
        batch_task = []
        for task_indices in indices:
            print(task_indices)
            support_indices = task_indices[0]
            query_indices = task_indices[1]

            support_set = [self.__getitem__(_) for _ in support_indices]
            query_set = [self.__getitem__(_) for _ in query_indices]

            batch_task.append([support_set, query_set])

        return batch_task
