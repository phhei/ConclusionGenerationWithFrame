import pathlib
from typing import List, Tuple, Union

import torch
from loguru import logger
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric, ReferenceFreeMetric
from transformers import BertForMaskedLM, BertTokenizer, BertConfig, BertForSequenceClassification

from Evaluation.GRUEN.Main import get_gruen


@Metric.register(name="GRUEN", exist_ok=False)
class GRUENMetric(ReferenceFreeMetric):
    def __init__(self):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Initializing GRUEN on {}", device)

        self.tokenizer_lm = BertTokenizer.from_pretrained("bert-base-cased")
        self.model_lm = BertForMaskedLM.from_pretrained("bert-base-cased").to(device)
        self.model_lm.eval()
        logger.trace("Load {}", self.model_lm)

        saved_pretrained_cola_model_dir = pathlib.Path("Evaluation", "GRUEN", "cola_model")
        self.cola_config = \
            BertConfig.from_pretrained(str(saved_pretrained_cola_model_dir.absolute()), num_labels=2,
                                       finetuning_task='CoLA')
        self.cola_tokenizer = \
            BertTokenizer.from_pretrained(str(saved_pretrained_cola_model_dir.absolute()), do_lower_case=0)
        self.cola_model = \
            BertForSequenceClassification.from_pretrained(str(saved_pretrained_cola_model_dir.absolute()),
                                                          from_tf=False, config=self.cola_config).to(device)
        self.cola_model.eval()
        logger.trace("Load {}", self.cola_model)

        self.include_premise = True

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                if isinstance(summary, List) or isinstance(summary, Tuple):
                    logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                                 summary[1])
                    summary = " ".join(map(lambda s: s if s.endswith(".") else "{}.".format(s.rstrip(" .!?")), summary))
                    logger.trace("Concatenated to: \"{}\"", summary)
                else:
                    logger.warning("Expected a list (premise, conclusion), but got only 1 part: \"{}\"", summary)
                cd = get_gruen(candidates=[summary], tokenizer_lm=self.tokenizer_lm, model_lm=self.model_lm,
                               cola_config=self.cola_config, cola_tokenizer=self.cola_tokenizer,
                               cola_model=self.cola_model)[0]
                d = {
                    "GRUEN": cd
                }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
