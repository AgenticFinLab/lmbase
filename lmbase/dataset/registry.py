"""
An interface to registry the datasets.
"""

import logging

data_factory = {
    "gsm8k": ("lmbase.dataset.gsm8k", "GSM8KDataset"),
    "math": ("lmbase.dataset.math", "MATHDataset"),
    "mmmu": ("lmbase.dataset.mmmu", "MMMUDataset"),
    "scienceqa": ("lmbase.dataset.scienceqa", "ScienceQADataset"),
    "aime2024": ("lmbase.dataset.aime2024", "AIME2024Dataset"),
    "aime2025": ("lmbase.dataset.aime2025", "AIME2025Dataset"),
    "aime19832024": ("lmbase.dataset.aime19832024", "AIME19832024Dataset"),
    "humaneval": ("lmbase.dataset.humaneval", "HumanEvalDataset"),
    "humanevalplus": ("lmbase.dataset.humanevalplus", "HumanEvalPlusDataset"),
    "codealpaca": ("lmbase.dataset.codealpaca", "CodeAlpacaDataset"),
    "hfcodealpaca": ("lmbase.dataset.hfcodealpaca", "CodeAlpacaDataset"),
    "theoremqa": ("lmbase.dataset.theoremqa", "TheoremQADataset"),
    "mathvision": ("lmbase.dataset.mathvision", "MathVisionDataset"),
    "mathvista": ("lmbase.dataset.mathvista", "MathVistaDataset"),
    "aokvqa": ("lmbase.dataset.aokvqa", "AOKVQADataset"),
    "vqav2": ("lmbase.dataset.vqav2", "VQAv2Dataset"),
    "mathverse": ("lmbase.dataset.mathverse", "MathVerseDataset"),
    "gqa": ("lmbase.dataset.gqa", "GQADataset"),
    "dapomath": ("lmbase.dataset.dapomath", "DAPOMathDataset"),
    "math500": ("lmbase.dataset.math500", "Math500Dataset"),
    "wemath": ("lmbase.dataset.wemath", "WeMathDataset"),
    "wemath2pro": ("lmbase.dataset.wemath2pro", "WeMath2ProDataset"),
    "geometry3k": ("lmbase.dataset.geometry3k", "Geometry3kDataset"),
    "mmlu": ("lmbase.dataset.mmlu", "MMLUDataset"),
    "gpqad": ("lmbase.dataset.gpqad", "GPQADiamondDataset"),
    "medqa": ("lmbase.dataset.medqa", "MedQADataset"),
    "arc": ("lmbase.dataset.arc", "ARCDataset"),
    "finagent": ("lmbase.dataset.finagent", "FinAgentDataset"),
    "financebench": ("lmbase.dataset.financebench", "FinanceBenchDataset"),
    "hotpotqa": ("lmbase.dataset.hotpotqa", "HotpotQADataset"),
    "multihoprag": ("lmbase.dataset.multihoprag", "MultiHopRAGDataset"),
    "concurrentqa": ("lmbase.dataset.concurrentqa", "ConcurrentQADataset"),
    "docfinqa": ("lmbase.dataset.docfinqa", "DocFinQADataset"),
    "finder": ("lmbase.dataset.finder", "FinDERDataset"),
    "narrativeqa": ("lmbase.dataset.narrativeqa", "NarrativeQADataset"),
    "stark": ("lmbase.dataset.stark", "STaRKDataset"),
    "vfinancebench": ("lmbase.dataset.vfinancebench", "VFinanceBenchDataset"),
    "gaia": ("lmbase.dataset.gaia", "GAIADataset"),
    "fincot": ("lmbase.dataset.fincot", "FinCoTDataset"),
    "codeforcescots": ("lmbase.dataset.codeforcescots", "CodeForcesCoTSDataset"),
    "multiarith": ("lmbase.dataset.multiarith", "MultiArithDataset"),
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
    "mathvista": "AI4Math/MathVista",
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
    "gpqad": "fingertap/GPQA-Diamond",
    "medqa": "openlifescienceai/medqa",
    "arc": "allenai/ai2_arc",
    "finagent": "vals-ai/finance_agent_benchmark",
    "financebench": "PatronusAI/financebench",
    "hotpotqa": "hotpotqa/hotpot_qa",
    "multihoprag": "yixuantt/MultiHopRAG",
    "concurrentqa": "stanfordnlp/concurrentqa-retrieval",
    "docfinqa": "kensho/DocFinQA",
    "finder": "Linq-AI-Research/FinDER",
    "narrativeqa": "meithnav/narrativeqa",
    "stark": "snap-stanford/stark",
    "vfinancebench": "virattt/financebench",
    "gaia": "gaia-benchmark/GAIA",
    "fincot": "TheFinAI/FinCoT",
    "codeforcescots": "open-r1/codeforces-cots",
    "multiarith": "ChilleD/MultiArith",
}


def get(config: dict, split="train"):
    """Get the dataset."""
    data_name = config["data_name"].lower()

    if data_name not in hf_datasets:
        raise KeyError(
            f"'{data_name}' - Unknown dataset. Available: {list(hf_datasets.keys())}"
        )

    hf_dataname = hf_datasets[data_name]
    logging.info("---> Loading %s data from %s (HF: %s)", split, data_name, hf_dataname)

    # Lazy import
    module_path, class_name = data_factory[data_name]
    module = __import__(module_path, fromlist=[class_name])
    dataset_class = getattr(module, class_name)

    dataset = dataset_class(split=split, hf_dataname=hf_dataname, config=config)
    logging.info("   - Obtained %s samples", len(dataset))
    return dataset
