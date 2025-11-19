## Guidance for the data loading 

> `lmbase` provides a unified interface for loading diverse datasets into a consistent format, as defined in `lmbase/dataset/base.py`. This format supports built-in sample types, such as "TextSample" or "TextCodeSample", or "VisualTextSample", or "VisualTextCodeSample" or user defined ones.

> The data sources are powered by the [Huggingface](https://huggingface.co/) library. Users can either download the datasets manually or let the code fetch them automatically from the Hugging Face Hub.

> All supported datasets are listed in `lmbase/dataset/registry.py`. For dataset-specific details, consult the comments in the respective data loader file.

> Please check the `examples/test_data` for the demo coding details.


### Loading datasets

One can easily load the data by using the following code:

1. For the `TextSample`

    ```console
    from lmbase.dataset import registry as dataset_registry

    loaded_dataset = dataset_registry.get(
            {
                "data_name": "gsm8k",
                "data_path": "EXPERIMENT/data",
            },
            "train",
        )
    ```

    After running the code above, the full dataset is cached in Hugging Face’s default `.cache` directory, while a small subset of samples is saved as a JSON file in `data_path` for quick verification.

    Besides, the `loaded_dataset` is the type derived from `torch.utils.data.Dataset`. Users can directly use it for training or evaluation.

2. For the `VisualTextSample`




### Loading datasets formally

To load the dataset to be the format required by the large models, that is `{"messages": [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "xxx"}]}`, users can follow two ways:

- Use the `formatter.map_sample` after loading the data
    ```console
    from lmbase import formatter

    xxx (Check the code above for the loading part)

    formatted_sample = formatter.map_sample(
        loaded_dataset[0],
        to_format="message",
    )
    ```

- Insert the `formatter.map_sample` into the data loading pipeline by setting `lm_format_function`.

    ```console
    from lmbase.dataset import registry as dataset_registry

    loaded_dataset = dataset_registry.get(
            {
                "data_name": "gsm8k",
                "data_path": "EXPERIMENT/data",
            },
            "train",
        )
    loaded_dataset.lm_format_function = lambda x: formatter.map_sample(x, to_format="message")
    formatted_sample = loaded_dataset[0]
    ```
