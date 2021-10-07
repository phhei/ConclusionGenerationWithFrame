from pathlib import Path
from typing import List, Union, Optional, Tuple

from loguru import logger
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric, ReferenceFreeMetric
from transformers import AutoTokenizer, PreTrainedTokenizer

import torch

from Frames import FrameSet
from FrameClassifier import FrameClassifier
from const import FRAME_START_TOKEN, FRAME_END_TOKEN, TOPIC_START_TOKEN, TOPIC_END_TOKEN


def get_frame_classifier(frame_set: FrameSet, preferred_model: str = "distilroberta-base",
                         corpus_data: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor, int]]]] = None,
                         token_frame_start: Optional[int] = None, token_frame_end: Optional[int] = None,
                         **kwargs) -> FrameClassifier:
    """
    Retrieves / loads a fine-tuned Frame-Classifier (gets a line of text, predict a frame class following the frame_set)

    :param frame_set: the desired frame-set
    :param preferred_model: a underlying transforming. Please don't define local paths here!
    :param corpus_data: data for training the underlying model. Is skipped if the model is already fine-tuned.
    :param token_frame_start: the id of the special token which is used to mark the beginning of a frame input
    :param token_frame_end: the id of the special token which is used to mark the end of a frame input
    :param kwargs:  - retrain: bool = a flag which forces a retraining even if the model is already fine-tuned
                    - token_topic_start: int
                    - topic_end_id: int
                    - all remaining kwargs are inputted to the Frame-Classifier. Have a look there for further
                    explanations
    :return: the frame classifier
    """
    def extract_span(seq: torch.Tensor, start_id: Optional[int], end_id: Optional[int]) -> str:
        logger.trace("Extract a part out of the tensor: {}", seq)
        seq: torch.Tensor = torch.squeeze(seq)
        if len(seq.shape) != 1:
            logger.error("Expected a single sample, hence a vector with exact dim, but get: {}", seq.shape)
            return "invalid"
        if start_id is None or end_id is None:
            logger.warning("One of the start or end ids are not given - infer the first tokens")
            return tokenizer.decode(token_ids=seq[1:3], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            start_index: torch.LongTensor = torch.where(seq == start_id)[0] + 1
            end_index: torch.LongTensor = torch.where(seq == end_id)[0]
            logger.trace("Retrieved following boundaries: {} -- {}", start_index, end_index)

            if len(start_index) == 0:
                logger.warning("The start index \"{}\" were not found in {}", start_index, seq)
                start_index_fn = 0
            elif len(start_index) >= 2:
                logger.info("There are multiple start positions: {} - take the first one: {}", start_index,
                            start_index[0])
                start_index_fn = start_index[0]
            else:
                logger.trace("Successfully found the entry point: {}", start_index)
                start_index_fn = start_index[0]
            if len(end_index) == 0:
                logger.warning("The end index \"{}\" were not found in {}", end_id, seq)
                end_index_fn = start_index_fn + 1
            elif len(end_index) >= 2:
                end_index_fn = torch.min(torch.where(end_index > start_index_fn, end_index, len(seq)))
                logger.info("There are multiple end positions: {} - take this one: {}", end_index, end_index_fn)
            else:
                logger.trace("Successfully found the end point: {}", end_index)
                end_index_fn = end_index[0]

            logger.trace("Determined following boundaries: [{}, {})", start_index_fn, end_index_fn)
            try:
                return tokenizer.decode(token_ids=seq[start_index_fn:end_index_fn], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
            except IndexError:
                logger.opt(exception=True).warning("Misinformed indices")
                return tokenizer.decode(token_ids=seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    model_path = Path("frame_classifier", str(frame_set), preferred_model.replace("/", "_"))

    if model_path.exists():
        logger.success("You desired model \"{}\" exists already :)", model_path.name)

        if "retrain" in kwargs:
            retrain = kwargs.pop("retrain")
        else:
            retrain = False
        if model_path.joinpath("metrics.txt").exists():
            logger.debug("Having following stats about the model (retrieved from {}):", model_path)
            logger.info(model_path.joinpath("metrics.txt").read_text(encoding="utf-8", errors="ignore"))
        else:
            logger.warning("Found no metrics-file... maybe something went wrong?")
            retrain = True
            logger.trace("Possibly retrain it...")

        model_path_fn = model_path
    else:
        logger.info("No model already trained... have to train it!")
        retrain = True
        model_path_fn = preferred_model

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(preferred_model)

    if retrain:
        if corpus_data is None:
            logger.warning("We're planning to fine-tune \"{}\", but you're giving no training data", preferred_model)
        else:
            logger.info("Having {} samples in total for fine-tine, eval and test the model", len(corpus_data))

        if "token_topic_start" in kwargs:
            topic_start_id = kwargs.pop("token_topic_start")
        else:
            topic_start_id = None
        if "topic_end_id" in kwargs:
            topic_end_id = kwargs.pop("token_topic_end")
        else:
            topic_end_id = None

        corpus_data = [
            point if isinstance(point, Tuple) else
            (point, frame_set.issues_specific_frame_to_generic(
                issue_specific_frame=extract_span(seq=point, start_id=token_frame_start, end_id=token_frame_end),
                topic=None if topic_start_id is None else extract_span(seq=point,
                                                                       start_id=topic_start_id, end_id=topic_end_id),
                fetch_column=None
            ))
            for point in corpus_data
        ]
    else:
        corpus_data = None

    return FrameClassifier(model=model_path_fn, frame_set=frame_set,
                           tokenizer=tokenizer, train_pairs=corpus_data, **kwargs)


@Metric.register(name="FrameScore", exist_ok=True)
class FrameScore(ReferenceFreeMetric):
    def __init__(self, frame_set: FrameSet, frame_classifier: FrameClassifier):
        super(FrameScore, self).__init__()

        self.frame_set = frame_set
        self.frame_classifier = frame_classifier
        self.include_premise = True

        logger.success("Initialized the {}-scorer: {}", frame_set, frame_classifier)
        logger.debug("--> type: {}", self.scorer.model_type)

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                             summary[1])
                predicted_frame_scores = self.frame_classifier.predict(sample=summary[1])
                logger.trace("\"{}\" is probably {} ({})", summary[1], predicted_frame_scores,
                             self.frame_set.data["label"])

                try:
                    if FRAME_START_TOKEN in summary[0] and FRAME_END_TOKEN in summary[0]:
                        expected_frame = \
                            summary[0][summary[0].index(FRAME_START_TOKEN)+1:summary[0].index(FRAME_START_TOKEN)]
                    else:
                        logger.warning("\"{}\" contains no information about the expected frame!", summary[0])
                        expected_frame = self.frame_set.name

                    if TOPIC_START_TOKEN in summary[0] and TOPIC_END_TOKEN in summary[0]:
                        topic = summary[0][summary[0].index(TOPIC_START_TOKEN)+1:summary[0].index(TOPIC_END_TOKEN)]
                    else:
                        logger.trace("\"{}\" contains no information about the current topic!", summary[0])
                        topic = None

                    expected_frame_id = self.frame_set.issues_specific_frame_to_generic(
                        issue_specific_frame=expected_frame,
                        topic=topic,
                        fetch_column=None,
                        semantic_reordering=False,
                        remove_stopwords=False
                    )

                    logger.trace("Expected frame: {}", expected_frame_id)

                    d = {
                        "framescore": {
                            "precision": predicted_frame_scores[expected_frame_id].item(),
                            "confidence": torch.max(predicted_frame_scores).item()
                        }
                    }
                    d["framescore"]["score"] = \
                        max(d["framescore"]["precision"],
                            min(2/3, .5*d["framescore"]["precision"]*(1/d["framescore"]["confidence"])))
                except IndexError:
                    logger.opt(exception=True).error("Error in determining the frame in the premise \"{}\" - "
                                                     "fall back to the default values", summary[0])
                    d = {
                        "framescore": {
                            "precision": .5,
                            "confidence": .5,
                            "score": .5
                        }
                    }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
