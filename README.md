# Overview
`lmbase` is a unified platform designed to support experiments with various reasoning methods for large models (LLMs, VLMs) in reasoning. This codebase is designed to be _easy to use_ and ensures fair comparisons across methods. It provides different modules that facilitate the implementation of new reasoning algorithms and their experiments on diverse datasets across various tasks for comprehensive evaluation.

> The folder named "EXPERIMENT" will be ignored by git by default. Thus, please always place experimental results under the "EXPERIMENT" folder.

> Please contribute reusable code by adding it to the relevant module or creating a new one if needed. 

## Code structure

The structure of `lmbase` is 

    .
    ├── configs                         # Configuration files
    ├── examples                        # Implemented examples
    ├── docs                            # Documentation
    ├── lmbase                          # CodeBase
    └──── datasets                      # Datasets 
    └──── utils                         # Useful functions


## Code guidance

Check the `docs` directory for more details about how to use each module of the codebase.
