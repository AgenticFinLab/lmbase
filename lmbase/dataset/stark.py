"""
Interface of the STaRK dataset.

STaRK (Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases) is a
large-scale benchmark for evaluating retrieval capabilities of LLMs on semi-structured
knowledge bases across diverse domains.

Dataset Source: https://huggingface.co/datasets/snap-stanford/stark
Documentation: https://stark.stanford.edu/

Description:
    A benchmark for evaluating LLM retrieval on textual and relational knowledge bases.
    It includes three knowledge bases across different domains:
    - Amazon: Product search knowledge base
    - MAG (Microsoft Academic Graph): Academic research knowledge base
    - Prime: Biomedical knowledge base

Size:
    - Amazon: ~XX,XXX queries (synthesized + human-generated)
    - MAG: ~XX,XXX queries (synthesized + human-generated)
    - Prime: ~XX,XXX queries (synthesized + human-generated)

Configurations:
    - amazon: Product search domain
    - mag: Academic research domain
    - prime: Biomedical domain
    Config setting in code: subset="amazon" or subset="mag" or subset="prime"

Splits:
    - synthesized_all_split: Main dataset with synthesized queries (~11,204 examples for Prime)
    - humen_generated_eval: Human-generated evaluation set (~98 examples for Prime)

Features:
    - query: The natural language query/question
    - query_id: Unique identifier for the query
    - answer_ids: List of answer node IDs
    - metadata: Additional information (removed in released version to prevent leakage)

License: Unknown - Please check the dataset page for license information.

Language: English

Paper: "STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases"
        https://arxiv.org/abs/2404.13207
"""

import logging
from datasets import load_dataset
from stark_qa import load_skb
from lmbase.dataset.base import TextSample, VisualTextBase


class STaRKDataset(VisualTextBase):
    """A consistent interface for the STaRK dataset with SKB support."""

    def map_dataset(self):
        """Map the dataset and load the corresponding SKB."""
        subset_name = self.config["subset"]
        config_map = {
            "amazon": "STaRK-Amazon",
            "mag": "STaRK-MAG",
            "prime": "STaRK-Prime",
        }
        config_name = config_map[subset_name]

        # 1. 加载 QA 数据集
        logging.info(f"   - 加载 QA 数据集: {config_name}, split: {self.split}")
        self.hf_dataset = load_dataset(self.hf_dataname, config_name, split=self.split)

        # 2. 加载对应的 SKB 知识库数据
        logging.info(f"   - 加载 SKB 知识库: {subset_name}")
        self.skb = load_skb(subset_name, download_processed=True)
        logging.info(f"   - SKB 加载完成: {self.skb is not None}")

        super().map_dataset()

    def to_format(self, sample):
        """Get the sample and enrich with SKB info if needed."""
        self.idx += 1
        query = sample["query"]
        answer_ids = sample["answer_ids"]

        # 将 ID 转换为字符串
        if isinstance(answer_ids, list):
            answer_str = ", ".join(str(aid) for aid in answer_ids)
        else:
            answer_str = str(answer_ids)

        question = f"{query}{self.SOLUTION_FORMAT_PROMPT}"

        return TextSample(
            main_id=f"ID{self.idx}",
            split=self.split,
            question=question,
            cot_answer=answer_str,
            groundtruth=answer_str,
            sample_info={
                "dataset": self.hf_dataname,
                "has_skb": self.skb is not None
            },
        )
