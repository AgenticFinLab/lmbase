## Guidance for the data loading 

> `lmbase` provides a unified interface for loading diverse datasets into a consistent format, as defined in `lmbase/dataset/base.py`. This format supports built-in sample types, such as "TextSample" or "TextCodeSample", or "VisualTextSample", or "VisualTextCodeSample" or user defined ones.

> The data sources are powered by the [Huggingface](https://huggingface.co/) library. Users can either download the datasets manually or let the code fetch them automatically from the Hugging Face Hub.


One can easily load the data by using the following code:

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




