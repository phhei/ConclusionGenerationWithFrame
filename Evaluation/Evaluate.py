import pathlib
import sqlite3
from typing import List, Tuple, Optional, Callable, Union, Dict

import numpy
import pandas
from loguru import logger
from sacrerouge.metrics import Metric, ReferenceBasedMetric, ReferenceFreeMetric
from Evaluation.Scores.Rougescore import RougeMetric
from Evaluation.Scores.BERTscore import BertScore, BertScorePremConc
from Evaluation.Scores.GRUENscore import GRUENMetric
from Evaluation.Scores.SurfaceScore import LengthScore, ClaimLikeScore
from Evaluation.Scores.FrameIdentifier import GenericFrameScore, get_generic_frame_classifier
from Frames import FrameSet


def score_sql(path: pathlib.Path, row_id: str, col_predict: str, col_ref: str, metrics: Optional[List[Metric]] = None,
              to_csv: bool = True):
    con = sqlite3.connect(database=str(path.absolute()))
    logger.info("Connect to database {}", con)
    sql = "SELECT {}, {}, {} FROM Predictions natural join Data".format(row_id, col_predict, col_ref)
    logger.debug("Fetch SQL-command: \"{}\"", sql)
    sql_ret = con.execute(sql).fetchall()
    logger.debug("Fetched {} samples", len(sql_ret))
    if len(sql_ret) >= 1:
        logger.trace("For example: predict = \"{}\", ground_truth = \"{}\"", sql_ret[0][1], sql_ret[0][2])
    else:
        logger.warning("\"{}\" is empty!", path)
    sql_ret_dict = {row_id: [i[0] for i in sql_ret],
                    col_predict: [i[1] for i in sql_ret],
                    col_ref: [i[2] for i in sql_ret]}

    if metrics is None or len(metrics) == 0:
        metrics = [BertScore(), RougeMetric()]
        logger.warning("You don't define any metrics. Therefore, we consider this one: {}",
                       " and ".join(map(lambda m: str(m), metrics)))

    for metric in metrics:
        logger.info("Let's evaluate \"{}\"", metric)

        scores = metric.score_all(summaries=sql_ret_dict[col_predict],
                                  references_list=[[r] for r in sql_ret_dict[col_ref]])
        scores_avg = metric.aggregate(scores)
        scores_avg = {"{}avg".format(k): v for k, v in scores_avg.items()}

        logger.success("Calculated {} scores. In average, we have: {}", len(scores), scores_avg)
        # {'bertscore': {'precision': 0.8754125714302063, 'recall': 0.8830074906349182, 'f1': 0.8790257215499878}}

        sql_ret_dict_update = pandas.json_normalize(data=scores, sep="_", errors="ignore").to_dict(orient="records")
        sql_ret_dict.update(pandas.DataFrame.from_records(data=sql_ret_dict_update).to_dict(orient="list"))
        logger.trace("Add {} scores to {} now", len(sql_ret_dict_update), sql_ret_dict)
        sql_ret_dict.update(
            pandas.json_normalize(data=scores_avg, sep="_", errors="ignore").to_dict(orient="records")[0]
        )
        logger.trace("Add averaged scores to {} now", sql_ret_dict)
        logger.debug("Resulting dict has now {} cols", len(sql_ret_dict))

    logger.success("Processed {} metrics", len(metrics))
    df = pandas.DataFrame.from_dict(data=sql_ret_dict, orient="columns")
    logger.debug("Resulting in following dataframe: {}", df)

    try:
        df.drop(columns=[col_predict, col_ref], inplace=False).to_sql(
            name="MetricScores", con=con, if_exists="replace", index=False, index_label=row_id
        )

        if to_csv:
            save_path_csv = path.parent.joinpath("{}_metric.csv".format(path.stem))
            logger.debug("OK, you want an additional CSV - you get it here: {}", save_path_csv.absolute())

            df.to_csv(
                path_or_buf=str(save_path_csv.absolute()), index=False, index_label=row_id, mode="a", encoding="utf-8"
            )

            logger.success("Saving done to {}", save_path_csv.name)
    except Exception:
        logger.opt(exception=True).error("Failed to write the SQL-stuff or the CSV")
    finally:
        con.close()


def score(predictions: List[Union[str, Tuple[str, str]]], reference: Optional[str], metrics: List[Tuple[Metric, str]],
          aggregation: Callable = numpy.average) -> int:
    logger.debug("OK, let's test {} predictions to the ground truth \"{}\" with {} metrics", len(predictions),
                 "n/a" if reference is None else reference, len(metrics))
    scores = []
    for prediction in predictions:
        scores.append(
            {
                "{}-{}".format(type(metric).__name__, number):
                    pandas.json_normalize(
                        data=metric.score(
                            summary=reference if isinstance(metric, ReferenceBasedMetric) else
                            (prediction if isinstance(prediction, str) else
                             (prediction if hasattr(metric, "include_premise") and metric.include_premise
                              else prediction[-1])),
                            references=None if isinstance(metric, ReferenceFreeMetric) else
                            [prediction if isinstance(prediction, str) else
                             (prediction if hasattr(metric, "include_premise") and metric.include_premise
                              else prediction[-1])]
                        ),
                        sep="_", errors="ignore").to_dict(orient="records")[0].get(metric_key, -1)
                for number, (metric, metric_key) in enumerate(metrics)
            }
        )
        logger.debug("The prediction \"{}\" has the following scores: {} (aggregated: {})", prediction, scores[-1],
                     aggregation(scores[-1].values()))

    logger.trace("Calculated the metric-scores for {} predictions", len(scores))

    scores_avg = [(i, aggregation(v.values())) for i, v in enumerate(scores)]
    scores_avg.sort(key=lambda s: s[-1], reverse=True)
    logger.trace(scores_avg)

    logger.info("The best prediction is the {}. one: \"{}\" with a score of {} ({})",
                scores_avg[0][0], predictions[scores_avg[0][0]], scores_avg[0][1],
                " # ".join(["{}: {}".format(m, round(v, 4)) for m, v in scores[scores_avg[0][0]].items()]))

    logger.trace("Return the list index of the best prediction...")
    return scores_avg[0][0]


def score_matrix(ret_dict: Dict[str, Dict[str, str]],
                 evaluation_instances: Optional[List[Metric]] = None) -> None:
    """
    Evaluates generated conclusions

    :param ret_dict: a dictionary of the form {sample_id: {input: ... predicted...}}.
                        Is generated by Trainer.generate-function
    :param evaluation_instances: The list of metrics which should evaluate each predicted conclusion
    :return: nothing - the ret-dict is updated (extended with the scores) in place
    """
    logger.info("Retrieved a premise-prediction-matrix out of {} samples", len(ret_dict))

    if evaluation_instances is None:
        logger.debug("No evaluation metrics given... take them all!")
        evaluation_instances = [
            BertScore(),
            BertScorePremConc(only_precision=False),
            GRUENMetric(),
            RougeMetric(),
            LengthScore(include_premise=True),
            ClaimLikeScore()
        ]
        logger.success("Having following evaluation instances now: {}",
                       " # ".join(map(lambda e: str(e), evaluation_instances)))
    else:
        logger.info("Retrieving {} evaluation instances", len(evaluation_instances))

    for sample_id, data in ret_dict.items():
        if sample_id == "columns":
            logger.trace("Skip - no sample")
        else:
            logger.debug("Evaluate sample \"{}\" now...", sample_id)
            additional_data = dict()
            for metric in evaluation_instances:
                if isinstance(metric, ReferenceBasedMetric):
                    premise = None
                else:
                    if hasattr(metric, "include_premise") and metric.include_premise:
                        key = metric.premise_col if hasattr(metric, "premise_col") else "input_without_special_tokens"
                        try:
                            premise = data.get(key, data["input"])
                        except KeyError:
                            logger.opt(exception=True).error("Sample \"{}\" is malformed. Expected \"{}\" or at least "
                                                             "\"input\" as premise key, but contains only: {}",
                                                             sample_id, key, ", ".join(data.keys()))
                            premise = ""
                    else:
                        logger.trace("Metric \"{}\" is referenceless, but doesn't require the premise", metric)
                        premise = None
                if isinstance(metric, ReferenceBasedMetric):
                    try:
                        ground_truth = data["ground_truth"]
                    except KeyError:
                        logger.opt(exception=True).warning("Problem by evaluating the metric \"{}\" - reference is "
                                                           "missing at sample \"{}\"", metric, sample_id)
                        ground_truth = ""
                else:
                    ground_truth = None

                for column, generated_conclusion \
                        in {k: v for k, v in data.items() if "prediction" in k and "debug" not in k}.items():
                    logger.debug("Let's investigate the the generated conclusion \"{}\"->\"{}\" of the sample \"{}\"",
                                 column, generated_conclusion, sample_id)
                    metric_dict = metric.score(
                        summary=ground_truth if ground_truth is not None else
                        (generated_conclusion if premise is None else (premise, generated_conclusion)),
                        references=([generated_conclusion] if premise is None else [(premise, generated_conclusion)])
                        if ground_truth is not None else None
                    )
                    for metric_key, metric_value in metric_dict.items():
                        logger.trace("Calculated \"{}\"", metric_key)
                        if isinstance(metric_value, int) or isinstance(metric_value, float):
                            logger.debug("Metric \"{}\" has a depth of 1: {}", metric_key, metric_value)
                            additional_data["{}_{}".format(column, metric_key)] = metric_value
                        elif isinstance(metric_value, Dict):
                            additional_data.update({
                                "{}_{}_{}".format(column, metric_key, k): v for k, v in metric_value.items()
                            })
                            logger.trace("Retrieved {} new measurements for \"{}\" -> \"{}\"", len(metric_value),
                                         sample_id, column)
                        else:
                            logger.warning("The Metric-dictionary has a unknown format: \"{}\" -> {}", metric_key,
                                           type(metric_value))
                logger.trace("Metric \"{}\" produces more scores (now: {})", len(additional_data))
            logger.success("Finished the evaluation of sample \"{}\". Retrieving {} scores out of {} metrics",
                           sample_id, len(additional_data), len(evaluation_instances))
            ret_dict[sample_id].update(additional_data)
    logger.success("Finished the matrix evaluation. Matrix has now a size of {}x{}", len(ret_dict),
                   max(map(lambda v: len(v), ret_dict.values())))


class CherryPicker:
    def __init__(self) -> None:
        self.metrics: List[Tuple[Metric, str]] = []
        self.weights: numpy.ndarray = numpy.empty(shape=(0,), dtype=float)
        self.weights_norm: numpy.ndarray = numpy.empty(shape=(0,), dtype=float)

    def add_metric(self, metric: Metric, metric_key: str, weight: float = 1.):
        logger.trace("OK, you want to add the metric \"{}\" (taking the key property \"{}\") with the weight of {}",
                     metric, metric_key, weight)
        self.metrics.append((metric, metric_key))
        logger.debug("Having {} metrics now...", len(self.metrics))

        if weight == 0:
            logger.error("The weight must not be 0,  set it to 1")
            weight = 1.

        self.weights = numpy.concatenate((self.weights, numpy.full(shape=(1, ), fill_value=weight, dtype=float)))
        self.weights_norm = self.weights / numpy.sum(self.weights)
        logger.debug("Updated weight vector: {} -> {}", self.weights, self.weights_norm)

    def cherry_picking(self, generated_sequences: List[Union[str, Tuple[str, str]]], reference: Optional[str])\
            -> Tuple[str, int]:
        if len(self.metrics) == 0:
            logger.warning("You didn't define any metrics until yet (please use \"add_metric\", "
                           "so we just return the first sequence")
            return generated_sequences[0] if isinstance(generated_sequences[0], str) else generated_sequences[0][-1], 0

        def picking_fn(x) -> float:
            if not isinstance(x, numpy.ndarray):
                x = numpy.fromiter(x, dtype=float)
            return numpy.sum(self.weights_norm * x)
        cherry_index = score(predictions=generated_sequences, reference=reference, metrics=self.metrics,
                             aggregation=picking_fn)
        logger.debug("Among {}, \"{}\" is the best one",
                     ", ".join(map(lambda s: "\"{}\"".format(s), generated_sequences)),
                     generated_sequences[cherry_index])
        return generated_sequences[cherry_index] \
                   if isinstance(generated_sequences[cherry_index], str) else generated_sequences[cherry_index][-1], \
               cherry_index

    def __str__(self) -> str:
        return "CherryPicker with {} ({})".format(
            ", ".join(map(lambda m: "{}[{}]".format(type(m[0]).__name__, m[1]), self.metrics)),
            self.weights_norm
        )

    def short_str(self) -> str:
        if len(self.metrics) == 0:
            return "empty"
        ret = ""
        for i, m in enumerate(self.metrics):
            ret += "{}_{}-{}".format(type(m[0]).__name__, m[1], round(self.weights_norm[i]*100))

        return ret


class CherryPickerSelection(dict):
    def __setitem__(self, k, v) -> None:
        if v is None:
            super().__setitem__(k, None)
            return

        cherry_picker = CherryPicker()

        metric_params = dict()
        for value in v:
            if isinstance(value, Metric):
                if len(metric_params) > 0:
                    cherry_picker.add_metric(**metric_params)
                    metric_params.clear()
                metric_params["metric"] = value
            elif isinstance(value, str):
                metric_params["metric_key"] = value
            elif isinstance(value, float) or isinstance(value, int):
                metric_params["weight"] = float(value)
            else:
                logger.warning("Detect strange param: {} (type: {})", value, type(value))
        if len(metric_params) > 0:
            cherry_picker.add_metric(**metric_params)

        super().__setitem__(k, cherry_picker)

    def __init__(self) -> None:
        super().__init__()

        self["None"] = None
        self["Empty"] = None
        self["SurfacePicker"] = None
        self["BERTPicker"] = None
        self["FramePicker"] = None
        self["ComprehensivePicker"] = None
        self["CheaterPicker"] = None

        logger.warning("We've preloaded following cherry-pickers: {}. Be aware! Until yet, all these pickers are "
                       "without any function. Please call load_cherrypicker_collection to change this!",
                       ", ".join(self.keys()))

    def load_cherrypicker_collection(self, **kwargs):
        if "used_frame_set" in kwargs:
            frame_set = kwargs.pop("used_frame_set")
            if frame_set is None or not isinstance(frame_set, FrameSet):
                logger.error("Your selected frame-set is {} but should be a FrameSet. "
                             "Ignore this error if you don't want to consider a specific generic frame set.",
                             type(frame_set))
                frame_set = None
                frame_classifier = None
                logger.warning("Will ignore all Cherry-Picker-components which consider the [generic] frame.")
            else:
                frame_classifier = get_generic_frame_classifier(frame_set=frame_set, **kwargs)
        else:
            frame_set = None
            frame_classifier = None

        self["None"] = None
        self["Empty"] = ()
        self["SurfacePicker"] = (GRUENMetric(), "GRUEN", 10,
                                 LengthScore(filter_stopwords=True, include_premise=True), "LengthScore_relative", 2,
                                 LengthScore(filter_stopwords=True), "LengthScore_content_word_ratio", 1,
                                 ClaimLikeScore(), "ClaimLikeScore_summary", 5)
        self["BERTPicker"] = (BertScorePremConc(), "bertscorePremCon")
        self["FramePicker"] = () if frame_set is None else (GenericFrameScore(frame_set=frame_set,
                                                                              frame_classifier=frame_classifier),
                                                            "framescore_score")
        comprehensive_picker = [GRUENMetric(), "GRUEN", 10,
                                LengthScore(filter_stopwords=True, include_premise=True), "LengthScore_relative", 2,
                                LengthScore(filter_stopwords=True), "LengthScore_content_word_ratio", 1,
                                ClaimLikeScore(), "ClaimLikeScore_summary", 5,
                                BertScorePremConc(), "bertscorePremCon", 12]
        if frame_set is not None or frame_classifier is not None:
            comprehensive_picker.extend([GenericFrameScore(frame_set=frame_set, frame_classifier=frame_classifier),
                                         "framescore_score", 15])
        self["ComprehensivePicker"] = tuple(comprehensive_picker)
        self["CheaterPicker"] = (RougeMetric(rouge_types=["rougeL"]), "rougeL", 1,
                                 BertScore(), "bertscore_f1", 2)
