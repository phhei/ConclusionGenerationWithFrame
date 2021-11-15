from pathlib import Path
from typing import List, Union

import torch
from loguru import logger
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric, ReferenceFreeMetric
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForSequenceClassification, PreTrainedModel

from Evaluation.Scores.Utils import clean_premise
from const import TOPIC_START_TOKEN, TOPIC_END_TOKEN


@Metric.register(name="StanceScore", exist_ok=True)
class StanceScore(ReferenceFreeMetric):
    def __init__(self, stance_classifier: Union[str, Path, PreTrainedModel],
                 classifier_tokenizer: Union[str, Path, PreTrainedTokenizer],
                 include_topic: bool = True):
        super(StanceScore, self).__init__()

        if isinstance(stance_classifier, Path):
            stance_classifier = str(stance_classifier.absolute())
        if isinstance(classifier_tokenizer, Path):
            classifier_tokenizer = str(classifier_tokenizer.absolute())

        try:
            self.stance_classifier = \
                AutoModelForSequenceClassification.from_pretrained(stance_classifier) \
                    if isinstance(stance_classifier, str) else stance_classifier
            self.classifier_tokenizer = \
                AutoTokenizer.from_pretrained(classifier_tokenizer) if isinstance(classifier_tokenizer, str) \
                    else classifier_tokenizer
            logger.info("Loaded following label mapping: {}", self.stance_classifier.config.id2label)
            if self.stance_classifier.config.num_labels != 3:
                logger.warning("The selected model is not a Seq2Class-model or has too few/ too much classes "
                               "(expected: 3 / actual: {}", self.stance_classifier.config.num_labels)
            else:
                if self.stance_classifier.config.label2id is None:
                    logger.warning("The stance classifier has no label2id-map - are you sure loading the right model?"
                                   "We will after-load it now...")
                    self.stance_classifier.config.id2label = {
                        "0": "CONTRADICTION",
                        "1": "NEUTRAL",
                        "2": "ENTAILMENT"
                    }
                    self.stance_classifier.config.label2id = {
                        "CONTRADICTION": 0,
                        "NEUTRAL": 1,
                        "ENTAILMENT": 2
                    }
                else:
                    logger.info("The stance classifier seems to be correctly configured: {}",
                                self.stance_classifier.config.label2id)
            if self.stance_classifier.training:
                logger.warning("\"{}\" is still in training mode - change this to inference!",
                               self.stance_classifier.config.model_type)
                self.stance_classifier.eval()
        except FileNotFoundError:
            logger.opt(exception=True).critical("You have to train this model first by executing \"NLIClassifier.py\"!")
            exit(hash(stance_classifier))

        self.include_topic = include_topic
        self.include_premise = True
        self.premise_col = "input"

        logger.success("Initialized the Stance-scorer: {}", self.stance_classifier.config.architectures)

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                             summary[1])

                topic = None
                if self.include_topic:
                    logger.trace("OK, let's retrieve the topic out of \"{}\"", summary[0])
                    try:
                        topic = summary[0][summary[0].index(TOPIC_START_TOKEN) + len(TOPIC_START_TOKEN):
                                           summary[0].index(TOPIC_END_TOKEN)].strip()
                    except ValueError:
                        logger.opt(exception=True).warning("Cannot retrieve the topic - leave it blank")

                processed_text = self.classifier_tokenizer(
                    text=clean_premise(summary[0]) if topic is None else "{}: {}".format(topic, clean_premise(summary[0])),
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

                try:
                    d = {"stancescore":
                        {
                            "precision":
                                predicted_stance_scores[self.stance_classifier.config.label2id["ENTAILMENT"]].item(),
                            "score":
                                predicted_stance_scores[self.stance_classifier.config.label2id["ENTAILMENT"]].item() *
                                (1 -
                                 predicted_stance_scores[self.stance_classifier.config.label2id["CONTRADICTION"]].item()**2)
                        }
                    }
                except KeyError:
                    logger.error("Your stance-classifier is misconfigured - should contain at least the label-keys "
                                 "\"ENTAILMENT\" and \"CONTRADICTION\"")
                    d = {"stancescore":
                        {
                            "precision": predicted_stance_scores[0].item(),
                            "score": predicted_stance_scores[0].item() * (1 - predicted_stance_scores[-1].item() ** 2)
                        }
                    }
                except IndexError:
                    logger.opt(exception=True).error("something went wrong - no chance calculating the StanceScore...")
                    d = {"stancescore":
                        {
                            "precision": 0.,
                            "score": 0.
                        }
                    }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
