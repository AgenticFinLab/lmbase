"""
An interface to registry the datasets.

The samples of each dataset within the lmbase are made in a consistent format.

Note that at the end of sample `question`, we add the solution flag prompt defined in `tmrl.identifier.py`. This is to prompt the model to add the final
solution within a specific identifier for the simple extraction.
"""

import logging
from lmbase.dataset import (
    gsm8k,
    math,
    aime2024,
    aime2025,
    aime19832024,
    theoremqa,
    mmmu,
    scienceqa,
    humaneval,
    humanevalplus,
    codealpaca,
    hfcodealpaca,
    mathvision,
    aokvqa,
    vqav2,
    mathverse,
    gqa,
    dapomath,
    math500,
    wemath,
    wemath2pro,
    geometry3k,
    mmlu,
)

data_factory = {
    "gsm8k": gsm8k.GSM8KDataset,
    "math": math.MATHDataset,
    "mmmu": mmmu.MMMUDataset,
    "scienceqa": scienceqa.ScienceQADataset,
    "aime2024": aime2024.AIME2024Dataset,
    "aime2025": aime2025.AIME2025Dataset,
    "aime19832024": aime19832024.AIME19832024Dataset,
    "humaneval": humaneval.HumanEvalDataset,
    "humanevalplus": humanevalplus.HumanEvalPlusDataset,
    "codealpaca": codealpaca.CodeAlpacaDataset,
    "hfcodealpaca": hfcodealpaca.CodeAlpacaDataset,
    "theoremqa": theoremqa.TheoremQADataset,
    "mathvision": mathvision.MathVisionDataset,
    "aokvqa": aokvqa.AOKVQADataset,
    "vqav2": vqav2.VQAv2Dataset,
    "mathverse": mathverse.MathVerseDataset,
    "gqa": gqa.GQADataset,
    "dapomath": dapomath.DAPOMathDataset,
    "math500": math500.Math500Dataset,
    "wemath": wemath.WeMathDataset,
    "wemath2pro": wemath2pro.WeMath2ProDataset,
    "geometry3k": geometry3k.Geometry3kDataset,
    "mmlu": mmlu.MMLUDataset,
}


hf_datasets = {
    "gsm8k": "openai/gsm8k",
    "math": "DigitalLearningGmbH/MATH-lighteval",
    "mmmu": "lmms-lab/MMMU",
    "scienceqa": "lmms-lab/ScienceQA",
    "aime2024": "HuggingFaceH4/aime_2024",
    "aime2025": "opencompass/AIME2025",
    "aime19832024": "di-zhang-fdu/AIME_1983_2024",
    "humaneval": "openai/openai_humaneval",
    "humanevalplus": "evalplus/humanevalplus",
    "codealpaca": "sahil2801/CodeAlpaca-20k",
    "hfcodealpaca": "HuggingFaceH4/CodeAlpaca_20K",
    "theoremqa": "TIGER-Lab/TheoremQA",
    "mathvision": "MathLLMs/MathVision",
    "aokvqa": "HuggingFaceM4/A-OKVQA",
    "vqav2": "lmms-lab/VQAv2",
    "mscoco": "bitmind/MS-COCO",
    "mathverse": "AI4Math/MathVerse",
    "gqa": "lmms-lab/GQA",
    "dapomath": "BytedTsinghua-SIA/DAPO-Math-17k",
    "math500": "HuggingFaceH4/MATH-500",
    "wemath": "We-Math/We-Math",
    "wemath2pro": "We-Math/We-Math2.0-Pro",
    "geometry3k": "hiyouga/geometry3k",
    "mmlu": "cais/mmlu",
}


def get(config: dict, split="train"):
    """Get the dataset."""

    data_name = config["data_name"].lower()
    hf_dataname = hf_datasets[data_name]
    logging.info(
        "---> Logging %s data from %s dataset linked to HF %s",
        split,
        data_name,
        hf_dataname,
    )
    dataset = data_factory[data_name](
        split=split, hf_dataname=hf_dataname, config=config
    )
    logging.info("   - Obtained %s samples", len(dataset))

    return dataset
