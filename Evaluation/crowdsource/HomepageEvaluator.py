import itertools
import math
import sys

import pandas
from pandas import DataFrame, read_csv
from pathlib import Path
import numpy

from loguru import logger
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

logger.remove()
logger.add(sys.stdout, level="INFO")

csv_sql_copy: Path = Path("../../.out/manual_evaluation/survey/Homepage-2 (cherry-picking)/result.csv")
relevant_cols = [
    "a_validity",
    "a_novelty",
    "a_issue_specific_frame",
    "a_generic_mapped_frame"
]
models_to_analyse: List[str] = [
    #"_random_conclusion",
    #"T5_ground_truth",
    #"T5_best_beam_prediction",
    #"T5+topic_best_beam_prediction",
    "T5smooth+topic_best_beam_prediction",
    "T5cherrysmooth+topic_selected_prediction",
    #"T5+topic+is_best_beam_prediction",
    "T5smooth+topic+is_best_beam_prediction",
    "T5cherrysmooth+topic+is_selected_prediction",
    "T5cherryfsmooth+topic+is_selected_prediction",
    "T5+topic+gen_best_beam_prediction",
    #"T5smooth+topic+gen_best_beam_prediction",
    "T5cherry+topic+gen_selected_prediction",
    "T5cherryf+topic+gen_selected_prediction",
    #"T5+topic+inf_best_beam_prediction",
    #"T5+topic+specific+generic_best_beam_prediction",
    "T5smooth+topic+specific+generic_best_beam_prediction",
    "T5cherrysmooth+topic+specific+generic_selected_prediction",
    "T5cherryfsmooth+topic+specific+generic_selected_prediction",
    #"T5+topic+specific+inf_best_beam_prediction",
    "T5smooth+topic+specific+inf_best_beam_prediction",
    "T5cherrysmooth+topic+specific+inf_selected_prediction",
    "T5cherryfsmooth+topic+specific+inf_selected_prediction"#,
    #"T5+topic+specific+generic+inf_best_beam_prediction"
]
comparative_models_to_analyse: List[Tuple[str, str]] = [
    #("T5+topic_best_beam_prediction", "T5smooth+topic_best_beam_prediction"),
    #("T5+topic+gen_best_beam_prediction", "T5smooth+topic+gen_best_beam_prediction"),
    #("T5+topic+specific+generic_best_beam_prediction", "T5smooth+topic+specific+generic_best_beam_prediction"),
    #("T5+topic+is_best_beam_prediction", "T5smooth+topic+is_best_beam_prediction"),
    #("T5+topic+specific+inf_best_beam_prediction", "T5smooth+topic+specific+inf_best_beam_prediction"),

    ("T5smooth+topic_best_beam_prediction", "T5cherrysmooth+topic_selected_prediction"),
    ("T5smooth+topic+is_best_beam_prediction", "T5cherrysmooth+topic+is_selected_prediction"),
    ("T5smooth+topic+is_best_beam_prediction", "T5cherryfsmooth+topic+is_selected_prediction"),
    ("T5cherrysmooth+topic+is_selected_prediction", "T5cherryfsmooth+topic+is_selected_prediction"),
    ("T5+topic+gen_best_beam_prediction", "T5cherry+topic+gen_selected_prediction"),
    ("T5+topic+gen_best_beam_prediction", "T5cherryf+topic+gen_selected_prediction"),
    ("T5cherry+topic+gen_selected_prediction", "T5cherryf+topic+gen_selected_prediction"),
    ("T5smooth+topic+specific+generic_best_beam_prediction", "T5cherrysmooth+topic+specific+generic_selected_prediction"),
    ("T5smooth+topic+specific+generic_best_beam_prediction", "T5cherryfsmooth+topic+specific+generic_selected_prediction"),
    ("T5cherrysmooth+topic+specific+generic_selected_prediction", "T5cherryfsmooth+topic+specific+generic_selected_prediction"),
    ("T5smooth+topic+specific+inf_best_beam_prediction", "T5cherrysmooth+topic+specific+inf_selected_prediction"),
    ("T5smooth+topic+specific+inf_best_beam_prediction", "T5cherryfsmooth+topic+specific+inf_selected_prediction"),
    ("T5cherrysmooth+topic+specific+inf_selected_prediction", "T5cherryfsmooth+topic+specific+inf_selected_prediction")
]
models_abbreviation: Dict[str, str] = {
    "_random_conclusion": "random",
    "T5_ground_truth": "ground_truth",
    "T5_best_beam_prediction": "T5-top",
    "T5+topic_best_beam_prediction": "T5",
    "T5+topic+is_best_beam_prediction": "T5+is",
    "T5+topic+gen_best_beam_prediction": "T5+gen",
    "T5+topic+inf_best_beam_prediction": "T5+inf",
    "T5+topic+specific+generic_best_beam_prediction": "T5+is+gen",
    "T5+topic+specific+inf_best_beam_prediction": "T5+is+inf",
    "T5+topic+specific+generic+inf_best_beam_prediction": "T5+is+gen+inf",
    "T5smooth_best_beam_prediction": "T5★-top",
    "T5smooth+topic_best_beam_prediction": "T5★",
    "T5cherrysmooth+topic_selected_prediction": "T5★°",
    "T5smooth+topic+is_best_beam_prediction": "T5★+is",
    "T5cherrysmooth+topic+is_selected_prediction": "T5★°+is",
    "T5cherryfsmooth+topic+is_selected_prediction": "T5★°°+is",
    "T5cherry+topic+gen_selected_prediction": "T5°+gen",
    "T5cherryf+topic+gen_selected_prediction": "T5°°+gen",
    "T5smooth+topic+gen_best_beam_prediction": "T5★+gen",
    "T5smooth+topic+inf_best_beam_prediction": "T5★+inf",
    "T5smooth+topic+specific+generic_best_beam_prediction": "T5★+is+gen",
    "T5cherrysmooth+topic+specific+generic_selected_prediction": "T5★°+is+gen",
    "T5cherryfsmooth+topic+specific+generic_selected_prediction": "T5★°°+is+gen",
    "T5smooth+topic+specific+inf_best_beam_prediction": "T5★+is+inf",
    "T5cherrysmooth+topic+specific+inf_selected_prediction": "T5★°+is+inf",
    "T5cherryfsmooth+topic+specific+inf_selected_prediction": "T5★°°+is+inf",
    "T5smooth+topic+specific+generic+inf_best_beam_prediction": "T5★+is+gen+inf"
}
models_sort: Dict[str, int] = {
    "random": 0,
    "ground_truth": 99,
    "T5-top": 10,
    "T5": 20,
    "T5+is": 30,
    "T5+gen": 40,
    "T5+inf": 50,
    "T5+is+gen": 60,
    "T5+is+inf": 70,
    "T5+is+gen+inf": 90,
    "T5★-top": 11,
    "T5°-top": 12,
    "T5★°-top": 16,
    "T5★": 21,
    "T5°": 22,
    "T5°°": 23,
    "T5★°": 26,
    "T5★°°": 27,
    "T5★+is": 31,
    "T5°+is": 32,
    "T5°°+is": 33,
    "T5★°+is": 36,
    "T5★°°+is": 37,
    "T5★+gen": 41,
    "T5°+gen": 42,
    "T5°°+gen": 43,
    "T5★°+gen": 46,
    "T5★°°+gen": 47,
    "T5★+inf": 51,
    "T5°+inf": 52,
    "T5°°+inf": 53,
    "T5★°+inf": 56,
    "T5★°°+inf": 57,
    "T5★+is+gen": 61,
    "T5°+is+gen": 62,
    "T5°°+is+gen": 63,
    "T5★°+is+gen": 66,
    "T5★°°+is+gen": 67,
    "T5★+is+inf": 71,
    "T5°+is+inf": 72,
    "T5°°+is+inf": 73,
    "T5★°+is+inf": 76,
    "T5★°°+is+inf": 77,
    "T5★+is+gen+inf": 91,
    "T5°+is+gen+inf": 92,
    "T5°°+is+gen+inf": 93,
    "T5★°+is+gen+inf": 96,
    "T5★°°+is+gen+inf": 97
}

stacked_bar_plot: bool = True


def fleiss_kappa(M: numpy.ndarray) -> float:
    """
    https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3

    Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(numpy.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = numpy.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = numpy.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (numpy.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = numpy.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)


def plot_dict(d: Dict):
    rows = math.ceil(len(d)/4)
    fig, axs = plt.subplots(rows, math.ceil(len(d)/rows), sharex=False, sharey=False)
    for number, name in enumerate(
            sorted([k for k in d.keys() if isinstance(d[k], Dict)],
                   key=lambda k: sum(map(lambda k_: models_sort.get(k_.strip(), -1), (ks := k.split("<"))))/len(ks))):
        lists = sorted(d[name].items())  # sorted by key, return a list of tuples

        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        axs[number % rows, math.floor(number/rows)].bar(x, height=y)
        for bar, value in enumerate(y):
            axs[number % rows, math.floor(number / rows)].text(bar - len(str(value))/15, value+1, str(value))
        axs[number % rows, math.floor(number/rows)].set_title(name)
        #axs[i%2][math.floor(i/2)].ylabel("count")
        axs[number % rows, math.floor(number/rows)].set_ylim([0, 30])
    plt.show()


def plot_dict_stacked_bars(d: Dict):
    x_labels = []
    y_labels = None
    data = None

    for model_name, stats in sorted([(k, v) for k, v in d.items() if isinstance(v, Dict)],
                                    key=lambda tpl: sum(map(lambda t_: models_sort.get(t_.strip(), -1),
                                                            (ks := tpl[0].split("<"))))/len(ks),
                                    reverse=False):
        logger.info("Plot the bar for model \"{}\" now", model_name)
        if "_nothing" in stats:
            logger.info("For our stacked bar-plots, we don't need the {} conclusion fulfilling no criterion",
                        stats.pop("_nothing"))

        lists = sorted(stats.items(), reverse=False)  # sorted by key, return a list of tuples

        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        x_labels.append(model_name)
        y_labels = y_labels or x
        if data is None:
            data = [[y_] for y_ in y]
        else:
            for e_, y_ in enumerate(y):
                data[e_].append(y_)

    for c_, criterion_scores in enumerate(data):
        bottoms = [sum([d[c__] for d in data[:c_]]) for c__ in range(len(criterion_scores))]
        plt.bar(x=x_labels,
                height=criterion_scores,
                width=1,
                bottom=bottoms,
                label=y_labels[c_],
                color=(.2, c_/len(data), .5)
                )

        for s, criterion_score in enumerate(criterion_scores):
            if criterion_score >= 1:
                plt.text(x=s,
                         y=bottoms[s]+criterion_score/2-.25,
                         s=criterion_score,
                         color="white")

    plt.legend(loc="upper left")
    plt.xticks(rotation=15, ha="right")
    plt.ylim([0, 30] if max(map(lambda c: sum(c), numpy.array(data, dtype=numpy.int).T.tolist())) <= 30 else [0, 60])
    plt.tight_layout()
    plt.show()


def calculate_inter_annotator_agreement(f_df: pandas.DataFrame) -> Dict[str, float]:
    ret = dict()
    for f_col in relevant_cols:
        agreement_matrix: numpy.ndarray = numpy.zeros(shape=(0, 3), dtype=int)
        for argument in set(f_df["argument_ID"]):
            logger.trace("Ok, let's have a look to the argument \"{}\"", argument)
            for variant_c1, variant_c2 in set(
                    [tuple(map(lambda r_: r_ if isinstance(r_, str) else None, r))
                     for _, r in f_df[f_df.argument_ID == argument][["conclusion_identifier_1", "conclusion_identifier_2"]].iterrows()]
            ):
                readable_key = variant_c1 if variant_c2 is None else "{} vs {}".format(variant_c1, variant_c2)
                if variant_c2 is None:
                    values = f_df[(f_df.argument_ID == argument) & (f_df.conclusion_identifier_1 == variant_c1)]
                    values = values.query(expr="`conclusion_identifier_2`.isna()", inplace=False)[f_col]
                else:
                    values = f_df[(f_df.argument_ID == argument) &
                                  (f_df.conclusion_identifier_1 == variant_c1) &
                                  (f_df.conclusion_identifier_2 == variant_c2)][f_col]
                if values.isna().any():
                    logger.debug("Ignore {}->{} because of unknown values related to \"{}\"",
                                 argument, readable_key, f_col)
                else:
                    if len(set(values)) == 1:
                        logger.trace("Cool, all {} annotators agree, that {}->{} is {}",
                                     len(values),
                                     argument, readable_key,
                                     "good/ PRO conclusion 2" if set(values).pop() == 1 else ("bad / PRO conclusion 1" if set(values).pop() == -1 else "ok/ equal"))
                    else:
                        logger.debug("There is a disagreement in {}->{}: {}", argument, readable_key, values)
                    agreement_matrix = \
                        numpy.append(arr=agreement_matrix,
                                     values=[(len(values[values == 1]),
                                              len(values[values == 0]),
                                              len(values[values == -1]))],
                                     axis=0)
        logger.info("For \"{}\", the general distribution is: good[C2]-{}/ok[o]-{}/bad[C1]-{}",
                    f_col,
                    numpy.sum(agreement_matrix[:, 0]), numpy.sum(agreement_matrix[:, 1]),
                    numpy.sum(agreement_matrix[:, 2]))
        kappa_agreement = fleiss_kappa(agreement_matrix)
        logger.success("For \"{}\", the fleiss-kappa is {}", f_col, kappa_agreement)
        ret[f_col] = kappa_agreement

    return ret


if __name__ == "__main__":
    df = read_csv(
        filepath_or_buffer=str(csv_sql_copy.absolute()),
        sep="|",
        index_col="a_ID",
        verbose=True
    )

    df_single = df.query("`conclusion_identifier_2`.isna()", inplace=False)
    df_comparative = df.query("`conclusion_identifier_2`.notna()", inplace=False)

    logger.info("Fetched {} single entries and {} comparative entries from {}",
                len(df_single), len(df_comparative), csv_sql_copy)

    annotators: List[int] = set(df["annotator_ID"])
    logger.debug("Found {} annotators: {}", len(annotators), " + ".join(map(lambda a: "\"{}\"".format(a), annotators)))

    if len(df_single) >= 3:
        logger.info("==> Inter-annotator-agreement for single-conclusion-votes:")
        calculate_inter_annotator_agreement(df_single)
    if len(df_comparative) >= 3:
        logger.info("==> Inter-annotator-agreement for comparative-conclusion-votes:")
        calculate_inter_annotator_agreement(df_comparative)

    logger.trace("Further analyses:")
    grouped_dfs: List[Tuple[DataFrame]] = []
    if len(df_single) >= 1:
        grouped_dfs.append(
            (
                "single",
                df_single.groupby(by=["argument_ID", "conclusion_identifier_1"],
                                  axis="index", as_index=True, sort=True, dropna=False).sum(numeric_only=True)
             )
        )
    if len(df_comparative) >= 1 and len(comparative_models_to_analyse) >= 1:
        grouped_dfs.append(
            (
                "comparative",
                df_comparative.groupby(by=["argument_ID", "conclusion_identifier_1", "conclusion_identifier_2"],
                                       axis="index", as_index=True, sort=True, dropna=False).sum(numeric_only=True)
            )
        )
    for mode, grouped_df in grouped_dfs:
        logger.debug("{}: {} groups", mode, len(grouped_df))
        stats_majority = {}
        stats_majority_only_positive = {}
        stats_full_agreement = {}
        stats_full_agreement_only_positive = {}

        for col in relevant_cols:
            stats_full_agreement["max_{}".format(col)] = grouped_df[col].max(skipna=True)
            stats_full_agreement_only_positive["max_{}".format(col)] = grouped_df[col].max(skipna=True)

        if mode == "single":
            iter_list = [(model, None, models_abbreviation[model]) for model in models_to_analyse]
        else:
            iter_list = [(model1, model2, "{} < {} ".format(models_abbreviation[model1], models_abbreviation[model2]))
                         for model1, model2 in comparative_models_to_analyse]

        for i in range(0, len(relevant_cols)+1):
            for t in itertools.combinations(relevant_cols, i):
                anti_set = set(relevant_cols).difference(t)
                for model1, model2, model_aberration in iter_list:
                    logger.debug("Let's analyse \"{}\" good in: {} (and bad in: {})",
                                 model_aberration, "-".join(t), "-".join(anti_set))

                    key = "+".join(
                        map(lambda t_: (t_[2:7]+t_[-3:] if len(t_) >= 12 else t_[2:]) if t_.startswith("a_") else t_, t)
                    ) if len(t) >= 1 else "_nothing"

                    filtered_df: DataFrame = \
                        grouped_df.loc[[index for index in grouped_df.index
                                        if index[1] == model1 and (model2 is None or index[2] == model2)]]
                    if model2 is not None:
                        filtered_df = filtered_df.append(
                            other=grouped_df.loc[[index for index in grouped_df.index
                                                  if index[1] == model2 and index[2] == model1]]*-1,
                            ignore_index=False,
                            verify_integrity=False, sort=False
                        )
                    if len(t) >= 1:
                        stats_majority_only_positive[model_aberration] = \
                            stats_majority_only_positive.get(model_aberration, dict())
                        stats_majority_only_positive[model_aberration][key] = \
                            len(filtered_df.query(expr=" and ".join(map(lambda t_: "`{}` > 0".format(t_), t)),
                                                  inplace=False))
                    stats_majority[model_aberration] = stats_majority.get(model_aberration, dict())
                    stats_majority[model_aberration][key] = len(
                        filtered_df.query(expr=" and ".join(
                            ([" and ".join(map(lambda t_: "`{}` > 0".format(t_), t))] if len(t) >= 1 else []) +
                            ([" and ".join(map(lambda t_: "`{}` <= 0".format(t_), anti_set))] if len(anti_set) >= 1 else [])
                        ), inplace=False)
                    )
                    if len(t) >= 1:
                        stats_full_agreement_only_positive[model_aberration] = \
                            stats_full_agreement_only_positive.get(model_aberration, dict())
                        stats_full_agreement_only_positive[model_aberration][key] = \
                            len(filtered_df.query(
                                expr=" and ".join(
                                    map(lambda t_: "`{}` == {}".format(t_, stats_full_agreement_only_positive["max_{}".format(col)]), t)
                                ),
                                inplace=False)
                            )
                    stats_full_agreement[model_aberration] = stats_full_agreement.get(model_aberration, dict())
                    stats_full_agreement[model_aberration][key] = len(
                        filtered_df.query(expr=" and ".join(
                            ([" and ".join(
                                map(lambda t_: "`{}` == {}".format(t_, stats_full_agreement["max_{}".format(col)]), t)
                            )] if len(t) >= 1 else []) +
                            ([" and ".join(
                                map(lambda t_: "(`{0}` {2}{1} or `{0}`.isna())".format(
                                    t_, stats_full_agreement["max_{}".format(col)], "< " if stacked_bar_plot else "== -"
                                ), anti_set)
                            )] if len(anti_set) >= 1 else [])
                        ), inplace=False)
                    )

        if stacked_bar_plot:
            plt.title("Full agreements ({})".format(mode))
            plot_dict_stacked_bars(stats_full_agreement)
            plt.title("Majority votes ({})".format(mode))
            plot_dict_stacked_bars(stats_majority)
        else:
            plt.title("Full agreements ({})".format(mode))
            plot_dict(stats_full_agreement)
            plt.title("Majority votes ({})".format(mode))
            plot_dict(stats_majority)