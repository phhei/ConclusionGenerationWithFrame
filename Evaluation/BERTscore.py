from typing import List, Union, Optional

from bert_score import BERTScorer
from loguru import logger
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType, ReferenceType
from sacrerouge.metrics import BertScore, Metric, ReferenceBasedMetric, ReferenceFreeMetric

import torch

from Evaluation.Evaluate import clean_premise


@Metric.register(name="bertscore", exist_ok=True)
class BertScore(ReferenceBasedMetric):
    def __init__(self, idf_sents: Optional[List[str]] = None, rescale_with_baseline: bool = True):
        super(BertScore, self).__init__()

        if idf_sents is not None:
            logger.warning("You enable the idf-weighting of the BERT-score ({} samples) - this is experimental!",
                           len(idf_sents))

        self.scorer = BERTScorer(
            model_type="microsoft/deberta-large-mnli",
            num_layers=18,
            idf=idf_sents is not None,
            idf_sents=idf_sents,
            batch_size=16,
            rescale_with_baseline=rescale_with_baseline,
            use_fast_tokenizer=False,
            lang="en"
        )

        logger.success("Initialized the BERT-scorer: {}", self.scorer.hash)
        logger.debug("--> type: {}", self.scorer.model_type)

    def score_multi_all(self, summaries_list: List[List[SummaryType]], references_list: List[List[ReferenceType]],
                        **kwargs) -> List[List[MetricsDict]]:
        """
        Scores a list of list summaries, where each inner list shares common references. All the summaries in the list
        summaries_list[i] should be scored using the references at references_list[i]

        :param summaries_list: the predicted conclusions
        :param references_list: the ground-truth-conclusions
        :param kwargs: n/a
        :return: The metric scores are stored in a MetricsDict, which is just a dictionary with some additional methods.
        The returned object should be a nested list that is parallel to the summaries in summaries_list.
        """
        metrics_lists = []
        for summaries, references in zip(summaries_list, references_list):
            metrics_list = []
            for summary in summaries:
                # Score this `summary` using `references`
                cd = self.scorer.score(refs=references, cands=[summary]*len(references),
                                       verbose=False, return_hash=False)
                d = {
                    "bertscore": {
                        "precision": torch.max(cd[0]).item() if isinstance(cd[0], torch.Tensor) else cd[0],
                        "recall": torch.max(cd[1]).item() if isinstance(cd[1], torch.Tensor) else cd[1],
                        "f1": torch.max(cd[2]).item() if isinstance(cd[2], torch.Tensor) else cd[2]
                    }
                }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists


@Metric.register(name="bertscorePremCon")
class BertScorePremConc(ReferenceFreeMetric):
    def __init__(self, idf_sents: List[str] = None, compute_novelty_of_conclusion: bool = True,
                 only_precision: bool = True, rescale_with_baseline: bool = True):
        super().__init__()
        if compute_novelty_of_conclusion and not only_precision:
            logger.info("You want to consider all scores of the BERT-score. This may be not appropriate since only the "
                        "precision reflect how much the conclusion (candidate) covers contents of the precision. "
                        "Hence, the novelty of a conclusion is determined by a low precision.")

        if idf_sents is not None:
            logger.warning("You enable the idf-weighting of the BERTscore ({} samples) - this is experimental!",
                           len(idf_sents))

        self.scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge",
            num_layers=18,
            idf=idf_sents is not None,
            idf_sents=idf_sents,
            batch_size=16,
            rescale_with_baseline=rescale_with_baseline,
            use_fast_tokenizer=False,
            lang="en"
        )

        logger.success("Initialized the BERT-scorer: {}", self.scorer.hash)
        logger.debug("--> type: {}", self.scorer.model_type)

        self.include_premise = True
        self.compute_novelty_of_conclusion = compute_novelty_of_conclusion
        self.only_precision = only_precision

    def score_multi_all(self, summaries_list: List[List[SummaryType]], **kwargs) -> List[List[MetricsDict]]:
        def switch_score(f_score: Union[float, torch.Tensor]) -> float:
            if self.compute_novelty_of_conclusion:
                f_score = 1-f_score
            return torch.max(f_score).item() if isinstance(f_score, torch.Tensor) else f_score

        metrics_lists = []
        for summaries in summaries_list:
            metrics_list = []
            for summary in summaries:
                logger.trace("Input consists of two parts: premise: \"{}\" --> conclusion: \"{}\"", summary[0],
                             summary[1])
                cd = self.scorer.score(refs=[clean_premise(summary[0])], cands=[summary[1]], verbose=False, return_hash=False)
                if self.only_precision:
                    d = {
                        "bertscorePremCon": switch_score(cd[0])
                    }
                else:
                    d = {
                        "bertscorePremCon": {
                            "precision": switch_score(cd[0]),
                            "recall": switch_score(cd[1]),
                            "f1": switch_score(cd[2])
                        }
                    }
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
