"""
An unit test for the data loading.
"""

import os
import json
import logging


from lmbase.dataset import registry as dataset_registry


if __name__ == "__main__":

    # Load the dataset
    loaded_dataset = dataset_registry.get(
        {
            "data_name": "gsm8k",
            "data_path": "EXPERIMENT/data",
        },
        "train",
    )

    # Convert the dataset to be the
