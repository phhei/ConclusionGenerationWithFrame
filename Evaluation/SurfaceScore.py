import abc
from math import exp
from typing import List, Optional, Union, Tuple

import nltk
from loguru import logger
from nltk import word_tokenize
from nltk.corpus import stopwords
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType
from sacrerouge.metrics import Metric, ReferenceFreeMetric


# related to https://aclanthology.org/E17-1017.pdf ON SURFACE


class SurfaceHint(ReferenceFreeMetric):
    tokenized_stuff = dict()
    try:
        stopwords_set = set(stopwords.words("english"))
    except OSError:
        logger.opt(exception=False).warning("We must download the stopwords first")
        if nltk.download("english"):
            sw = set(stopwords.words("english"))
            logger.success("Downloaded the English-nltk-package and found {} stopwords", len(sw))
            stopwords_set = sw
        else:
            logger.error("Were not able to download the stopwords - proceed with an empty list!")
            stopwords_set = set()

    def __init__(self, filter_stopwords: Optional[bool] = None):
        super().__init__()

        self.filter_stopwords = filter_stopwords

    def preprocess_input(self, sample: str) -> List[Union[Tuple[str, bool], str]]:
        if sample in SurfaceHint.tokenized_stuff:
            sample_processed = SurfaceHint.tokenized_stuff[sample]
            logger.info("Sample \"{}\" is already preprocessed: {}", sample, sample_processed)
            if self.filter_stopwords is True and len(sample_processed) >= 1 and isinstance(sample_processed[0], str):
                logger.warning("Didn't compute the stopwords until yet - so let's do it!")
                SurfaceHint.tokenized_stuff[sample] = \
                    [(t, t.lower() in SurfaceHint.stopwords_set) for t in sample_processed]
                return SurfaceHint.tokenized_stuff[sample]
            elif self.filter_stopwords is False and len(sample_processed) >= 1 and isinstance(sample_processed[0], str):
                return [(t, None) for t in sample_processed]
            elif self.filter_stopwords is None and len(sample_processed) >= 1 and isinstance(sample_processed[0], Tuple):
                return [t[0] for t in sample_processed]
            return sample_processed
        else:
            logger.debug("We have to preprocess \"{}\" first.", sample)
            SurfaceHint.tokenized_stuff[sample] = \
                [t if self.filter_stopwords is None else
                 (t, (t.lower() in SurfaceHint.stopwords_set) if self.filter_stopwords else None)
                 for t in word_tokenize(sample)]
            logger.trace("Finished! Integrated a list of {} tokens into a dictionary of {} samples",
                         len(SurfaceHint.tokenized_stuff[sample]), len(SurfaceHint.tokenized_stuff)-1)
            return SurfaceHint.tokenized_stuff[sample]

    def return_length(self, preprocessed_sample:  List[Union[Tuple[str, bool], str]],
                      including_stopwords: Optional[bool] = None) -> int:
        if including_stopwords is None:
            if self.filter_stopwords is None or self.filter_stopwords is False:
                including_stopwords = True
            else:
                including_stopwords = False
            logger.debug("[Inferred] Including stopwords: {}", including_stopwords)

        if not including_stopwords and self.filter_stopwords is None:
            logger.warning("Can't ignore stopwords when you're not willing to detect them!")
            return 0

        if including_stopwords:
            return len(preprocessed_sample)
        else:
            return len([t[0] for t in preprocessed_sample if not t[1]])

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        logger.warning("{} has to be implemented by a subclass!", self)
        return super().score_multi_all(summaries_list, **kwargs)


@Metric.register(name="LengthScore")
class LengthScore(SurfaceHint):
    """
    The key points from graph-based summarization model are relatively longer. This also improves their informativeness,
    matching findings of \newcite{syed:2021}.
    """

    def __init__(self, filter_stopwords: Optional[bool] = None, include_premise: bool = False):
        super().__init__(filter_stopwords)

        self.include_premise = include_premise

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                if isinstance(summary, List) or isinstance(summary, Tuple):
                    logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                                 summary[1])
                    preprocessed_premise = self.preprocess_input(summary[0])
                    preprocessed_conclusion = self.preprocess_input(summary[1])
                else:
                    if self.include_premise:
                        logger.warning("Expected a list (premise, conclusion), but got only 1 part: \"{}\"", summary)
                    preprocessed_premise = None
                    preprocessed_conclusion = self.preprocess_input(summary)

                d = {
                    "LengthScore": {
                        "absolute": 1/(1+exp(
                            -(self.return_length(preprocessed_sample=preprocessed_conclusion, including_stopwords=True)/3.-4)
                        )),
                        "absolute_non_stopwords":  1/(1+exp(
                            -(self.return_length(preprocessed_sample=preprocessed_conclusion, including_stopwords=False)/2.5-4)
                        ))
                    }
                }
                if preprocessed_premise is not None:
                    d["LengthScore"]["relative"] = \
                        d["LengthScore"]["absolute"]*min(
                            10.,
                            self.return_length(preprocessed_sample=preprocessed_premise)/
                            max(1, self.return_length(preprocessed_sample=preprocessed_conclusion))
                        )/10.
                if self.filter_stopwords:
                    d["LengthScore"]["content_word_ratio"] = \
                        self.return_length(preprocessed_sample=preprocessed_conclusion, including_stopwords=False)/\
                        max(1, self.return_length(preprocessed_sample=preprocessed_conclusion,
                                                  including_stopwords=True))
                logger.debug("Calculated the surface scores for \"{}\" ({}): {}", summary, len(d["LengthScore"]),
                             " | ".join(map(lambda kv: "{}: {}".format(kv[0], round(kv[1], 2)),
                                            d["LengthScore"].items())))
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists


@Metric.register(name="ClaimLikeScore")
class ClaimLikeScore(SurfaceHint):
    def __init__(self):
        super().__init__(filter_stopwords=None)

        self.include_premise = False

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                if isinstance(summary, List) or isinstance(summary, Tuple):
                    logger.error("{} expected a single conclusion, but you gave (probably) premise and conclusion!",
                                 self)
                    summary = summary[-1]
                preprocessed_summary = self.preprocess_input(sample=summary)
                # ./Promotion/Meetings/_mit Doktorvater/Doktortalk_20210128.pdf
                d = {
                    "ClaimLikeScore": {
                        "include_STATE": int("is" in preprocessed_summary or "are" in preprocessed_summary
                                             or "was" in preprocessed_summary or "were" in preprocessed_summary),
                        "include_comparison": int(("is" in preprocessed_summary or "are" in preprocessed_summary)
                                                  and
                                                  ("better" in preprocessed_summary or "worse" in preprocessed_summary)
                                                  and
                                                  "than" in preprocessed_summary),
                        "include_normative": int("should" in preprocessed_summary or "do" in preprocessed_summary
                                                 or "did" in preprocessed_summary or "can" in preprocessed_summary
                                                 or "have" in preprocessed_summary or "has" in preprocessed_summary),
                        "include_reason": int("cause" in preprocessed_summary or "caused" in preprocessed_summary
                                              or "reason" in preprocessed_summary)
                    }
                }
                d["ClaimLikeScore"]["summary"] = sum(d["ClaimLikeScore"].values())/len(d["ClaimLikeScore"].values())
                logger.trace("Calculated {} different surface scores. The result is: on the first glance, "
                             "the conclusion candidate is claim-like with a probability of {}%",
                             len(d["ClaimLikeScore"]), round(d["ClaimLikeScore"]["summary"]*100.))
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
