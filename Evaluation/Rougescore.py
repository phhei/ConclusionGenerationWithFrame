from typing import List, Optional

from rouge_score import rouge_scorer
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType, ReferenceType
from sacrerouge.metrics import Metric, ReferenceBasedMetric


@Metric.register('RougeMetric')
class RougeMetric(ReferenceBasedMetric):
    def __init__(self, rouge_types: Optional[List[str]] = None, only_use_f_score: bool = True):
        super().__init__()
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        self.only_use_f_score = only_use_f_score

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
                d = {}
                for reference in references:
                    cd = self.rouge_scorer.score(target=reference, prediction=summary)
                    for k, v in cd.items():
                        if self.only_use_f_score:
                            d[k] = max(d.get(k, 0.), v.fmeasure)
                        else:
                            d[k] = d.get(k, {"precision": 0., "recall": 0., "f1": 0.})
                            d[k]["precision"] = max(d[k]["precision"], v.precision)
                            d[k]["recall"] = max(d[k]["recall"], v.recall)
                            d[k]["f1"] = max(d[k]["f1"], v.fmeasure)
                metrics_list.append(MetricsDict(d))
            metrics_lists.append(metrics_list)
        return metrics_lists
