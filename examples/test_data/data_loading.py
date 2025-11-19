"""
An unit test for the data loading.
"""

from lmbase.dataset import registry as dataset_registry

from lmbase import formatter

if __name__ == "__main__":

    # Load the dataset
    loaded_dataset = dataset_registry.get(
        {
            "data_name": "gsm8k",
            "data_path": "EXPERIMENT/data",
        },
        "train",
    )
    print(loaded_dataset)
    print(loaded_dataset[0])
    # 1. Convert the dataset to be the format required by the LMs
    formatted_sample = formatter.map_sample(
        loaded_dataset[0],
        to_format="message",
    )
    print(formatted_sample)
    # 2. Use the `lm_format_function` to format the dataset
    loaded_dataset.lm_format_function = lambda x: formatter.map_sample(
        x, to_format="message"
    )
    print(loaded_dataset[0])
