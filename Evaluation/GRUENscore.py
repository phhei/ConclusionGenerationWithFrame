import pathlib
from typing import List, Tuple, Union

import nltk
import torch
from loguru import logger
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric, ReferenceFreeMetric
from transformers import BertForMaskedLM, BertTokenizer, BertConfig, BertForSequenceClassification

from Evaluation.Utils import clean_premise
from Evaluation.GRUEN.Main import get_gruen
from const import FRAME_END_TOKEN, TOPIC_END_TOKEN


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
        self.premise_col = "input_without_special_tokens"

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                if isinstance(summary, List) or isinstance(summary, Tuple):
                    logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                                 summary[1])
                    summary_prem = summary[0] if summary[0].endswith(".") else "{}.".format(summary[0].rstrip(" .!?"))
                    summary_concl = summary[1]
                    summary = "{} {}".format(summary_prem, summary_concl)
                    logger.trace("Concatenated to: \"{}\"", summary)
                else:
                    logger.warning("Expected a list (premise, conclusion), but got only 1 part: \"{}\"", summary)
                while True:
                    try:
                        cd = get_gruen(candidates=[summary], tokenizer_lm=self.tokenizer_lm, model_lm=self.model_lm,
                                       cola_config=self.cola_config, cola_tokenizer=self.cola_tokenizer,
                                       cola_model=self.cola_model)[0]
                        break
                    except LookupError:
                        logger.opt(exception=False).warning("You have possibly not downloaded required python-packages."
                                                            " Let's change this!")
                        if nltk.download("punkt"):
                            logger.success("OK, downloaded the missing part - try it again")
                        else:
                            logger.error("We're not able to download the missing dependency - we're sorry.")
                            cd = 0.
                            break
                    except Exception:
                        logger.opt(exception=True).critical("Problemns with the GRUEN-library")
                        cd = 0.
                        break
                d = {
                    "GRUEN": cd
                }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
