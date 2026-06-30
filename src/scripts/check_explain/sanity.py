import copy
import torch
import numpy as np
from tqdm import tqdm



def sanity_check(
    explainer, test_loader, T, scale="all"):
    test_loader_pbar = tqdm(
        test_loader, desc="Explaining", position=0, leave=True, unit="boT"
    )
    pdas = []
    ndas = []
    combines = []

    for metabatch_id, boT in enumerate(test_loader_pbar):
        boT_pbar = tqdm(
            boT, desc=f"Batch {metabatch_id}", position=1, leave=False, unit="task"
        )
        for task_id, (support, query) in enumerate(boT_pbar):
            sup_x, sup_y = support
            que_x, que_y = query

            
