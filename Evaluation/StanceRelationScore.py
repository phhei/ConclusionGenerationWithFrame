from pathlib import Path
from typing import List, Union, Optional, Tuple

from loguru import logger
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric, ReferenceFreeMetric
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel, PreTrainedModel

import torch

from Evaluation.Evaluate import clean_premise


@Metric.register(name="StanceScore", exist_ok=True)
class StanceScore(ReferenceFreeMetric):
    def __init__(self, stance_classifier: Union[str, Path, PreTrainedModel],
                 classifier_tokenizer: Union[str, Path, PreTrainedTokenizer]):
        super(StanceScore, self).__init__()

        if isinstance(stance_classifier, Path):
            stance_classifier = str(stance_classifier.absolute())
        if isinstance(classifier_tokenizer, Path):
            classifier_tokenizer = str(classifier_tokenizer.absolute())

        try:
            self.stance_classifier = \
                AutoModel.from_pretrained(stance_classifier) if isinstance(stance_classifier, str) else stance_classifier
            self.classifier_tokenizer = \
                AutoTokenizer.from_pretrained(classifier_tokenizer) if isinstance(classifier_tokenizer, str) \
                    else classifier_tokenizer
            logger.info("Loaded following label mapping: {}", self.stance_classifier.config.id2label)
            if self.stance_classifier.config.num_labels != 3:
                logger.warning("The selected model is not a Seq2Class-model or has too few/ too much classes "
                               "(expected: 3 / actual: {}", self.stance_classifier.config.num_labels)
            if self.stance_classifier.training:
                logger.warning("\"{}\" is still in training mode - change this to inference!",
                               self.stance_classifier.config.model_type)
                self.stance_classifier.eval()
        except FileNotFoundError:
            logger.opt(exception=True).critical("You have to train this model first by executing \"NLIClassifier.py\"!")
            exit(hash(stance_classifier))

        self.include_premise = True

        logger.success("Initialized the Stance-scorer: {}", self.stance_classifier.config.architectures)

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                             summary[1])
                processed_text = self.classifier_tokenizer(
                    text=clean_premise(summary[0]),
                    text_pair=summary[1],
                    add_special_tokens=True,
                    is_split_into_words=False,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    return_offsets_mapping=False,
                    return_length=False,
                    verbose=False
                )
                predicted_stance_scores = torch.softmax(
                    input=torch.squeeze(
                        input=self.stance_classifier(
                            **processed_text,
                            return_dict=True
                        )["logits"]),
                    dim=-1)
                logger.trace("\"{}\" is probably {} ({})", predicted_stance_scores,
                             self.stance_classifier.config.id2label)

                d = {"stancescore": {
                    "precision": predicted_stance_scores[-1].item(),
                    "score": predicted_stance_scores[-1].item()*(1-predicted_stance_scores[0].item()**2)
                }
                }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists