# Overview
`lmbase` is a unified platform designed to support experiments with various reasoning methods for large models (LLMs, VLMs) in reasoning. This codebase is designed to be _easy to use_ and ensures fair comparisons across methods. It provides different modules that facilitate the implementation of new reasoning algorithms and their experiments on diverse datasets across various tasks for comprehensive evaluation.

> The folder named "EXPERIMENT" will be ignored by git by default. Thus, please always place experimental results under the "EXPERIMENT" folder.

> Please contribute reusable code by adding it to the relevant module or creating a new one if needed. 

## Code structure

The structure of `lmbase` is 

    .
    ├── configs                         # Configuration files
    ├── examples                        # Example scripts and usage (for users)
    ├── docs                            # Documentation
    ├── lmbase                          # Core library
    │   ├── datasets                    # Built-in datasets and loaders
    │   ├── inference                   # Inference APIs and wrappers
    │   └── utils                       # Utility functions


## Code guidance

Check the `docs` directory for more details about how to use each module of the codebase.


## Using the `.env.example` File

This repository provides a `.env.example` file to help you set up your environment variables required for running code (such as API keys and custom endpoints). To get started:

1. **Copy** the `.env.example` file and rename it to `.env` in the root directory of your project:
   ```
   cp .env.example .env
   ```

2. **Edit** the `.env` file and fill in your own credentials and variable values as needed.

3. Many scripts (such as those in the `examples/` directory) will automatically load environment variables using [python-dotenv](https://github.com/theskumar/python-dotenv) or similar libraries.

> **Note:** Never commit actual credentials or secrets to version control—`.env.example` contains only placeholder values for safety.
