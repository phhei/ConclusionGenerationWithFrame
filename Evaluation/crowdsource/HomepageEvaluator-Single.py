import itertools
import math

from pandas import DataFrame, read_csv
from pathlib import Path
import numpy

from loguru import logger
from typing import List, Dict

import matplotlib.pyplot as plt

csv_sql_copy: Path = Path("../../.out/manual_evaluation/survey/Homepage-0/result.csv")
relevant_cols = [
    "a_validity",
    "a_novelty",
    "a_issue_specific_frame",
    "a_generic_mapped_frame"
]
models_to_analyse = ["_random_conclusion",
                          "T5_ground_truth", "T5_best_beam_prediction",
                          "T5+topic_best_beam_prediction",
                          "T5+topic+is_best_beam_prediction",
                          "T5+topic+gen_best_beam_prediction",
                          "T5+topic+inf_best_beam_prediction",
                          "T5+topic+specific+generic_best_beam_prediction",
                          "T5+topic+specific+inf_best_beam_prediction",
                          "T5+topic+specific+generic+inf_best_beam_prediction"]


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
    for number, name in enumerate([k for k in d.keys() if isinstance(d[k], Dict)]):
        lists = sorted(d[name].items())  # sorted by key, return a list of tuples

        x, y = zip(*lists)  # unpack a list of pairs into two tuples

        axs[number % rows, math.floor(number/rows)].bar(x, height=y)
        for bar, value in enumerate(y):
            axs[number % rows, math.floor(number / rows)].text(bar - len(str(value))/15, value+1, str(value))
        axs[number % rows, math.floor(number/rows)].set_title(name)
        #axs[i%2][math.floor(i/2)].ylabel("count")
        axs[number % rows, math.floor(number/rows)].set_ylim([0, 30])
    plt.show()


if __name__ == "__main__":
    df = read_csv(
        filepath_or_buffer=str(csv_sql_copy.absolute()),
        sep="|",
        index_col="a_ID",
        verbose=True
    )

    logger.info("Fetched {} entries from {}", len(df), csv_sql_copy)

    annotators: List[int] = set(df["annotator_ID"])
    logger.debug("Found {} annotators: {}", len(annotators), " + ".join(map(lambda a: "\"{}\"".format(a), annotators)))

    for col in relevant_cols:
        agreement_matrix: numpy.ndarray = numpy.zeros(shape=(0, 3), dtype=int)
        for argument in set(df["argument_ID"]):
            logger.trace("Ok, let's have a look to the argument \"{}\"", argument)
            for variant in set(df["conclusion_identifier_1"]):
                values = df[df.argument_ID == argument][df.conclusion_identifier_1 == variant][col]
                if values.isna().any():
                    logger.debug("Ignore {}->{} because of unknown values related to \"{}\"", argument, variant, col)
                else:
                    if len(set(values)) == 1:
                        logger.trace("Cool, all {} annotators agree, that {}->{} is {}",
                                     len(values),
                                     argument, variant,
                                     "good" if set(values).pop() == 1 else ("bad" if set(values).pop() == -1 else "ok"))
                    else:
                        logger.debug("There is a disagreement in {}->{}: {}", argument, variant, values)
                    agreement_matrix = \
                        numpy.append(arr=agreement_matrix,
                                     values=[(len(values[values == 1]),
                                              len(values[values == 0]),
                                              len(values[values == -1]))],
                                     axis=0)
        logger.info("For \"{}\", the general distribution is: good-{}/ok-{}/bad-{}",
                    col,
                    numpy.sum(agreement_matrix[:, 0]), numpy.sum(agreement_matrix[:, 1]),
                    numpy.sum(agreement_matrix[:, 2]))
        logger.success("For \"{}\", the fleiss-kappa is {}", col, fleiss_kappa(agreement_matrix))

    logger.trace("Further analyses:")
    grouped_df: DataFrame = df.groupby(by=["argument_ID", "conclusion_identifier_1"],
                                       axis="index", as_index=True, sort=True, dropna=False).sum(numeric_only=True)
    logger.debug("{} groups", len(grouped_df))
    stats_majority = {}
    stats_majority_only_positive = {}
    stats_full_agreement = {}
    stats_full_agreement_only_positive = {}

    for col in relevant_cols:
        stats_full_agreement["max_{}".format(col)] = grouped_df[col].max(skipna=True)
        stats_full_agreement_only_positive["max_{}".format(col)] = grouped_df[col].max(skipna=True)

    for i in range(0, len(relevant_cols)+1):
        for t in itertools.combinations(relevant_cols, i):
            anti_set = set(relevant_cols).difference(t)
            for model in models_to_analyse:
                logger.debug("Let's analyse \"{}\" good in: {} (and bad in: {})",
                             model, "-".join(t), "-".join(anti_set))

                key = "+".join(
                    map(lambda t_: (t_[2:7]+t_[-3:] if len(t_) >= 12 else t_[2:]) if t_.startswith("a_") else t_, t)
                ) if len(t) >= 1 else "_nothing"

                filtered_df: DataFrame = \
                    grouped_df.loc[[index for index in grouped_df.index if index[1] == model]]
                if len(t) >= 1:
                    stats_majority_only_positive[model] = stats_majority_only_positive.get(model, dict())
                    stats_majority_only_positive[model][key] = \
                        len(filtered_df.query(expr=" and ".join(map(lambda t_: "`{}` > 0".format(t_), t)),
                                              inplace=False))
                stats_majority[model] = stats_majority.get(model, dict())
                stats_majority[model][key] = len(
                    filtered_df.query(expr=" and ".join(
                        ([" and ".join(map(lambda t_: "`{}` > 0".format(t_), t))] if len(t) >= 1 else []) +
                        ([" and ".join(map(lambda t_: "`{}` <= 0".format(t_), anti_set))] if len(anti_set) >= 1 else [])
                    ), inplace=False)
                )
                if len(t) >= 1:
                    stats_full_agreement_only_positive[model] = stats_full_agreement_only_positive.get(model, dict())
                    stats_full_agreement_only_positive[model][key] = \
                        len(filtered_df.query(
                            expr=" and ".join(
                                map(lambda t_: "`{}` == {}".format(t_, stats_full_agreement_only_positive["max_{}".format(col)]), t)
                            ),
                            inplace=False)
                        )
                stats_full_agreement[model] = stats_full_agreement.get(model, dict())
                stats_full_agreement[model][key] = len(
                    filtered_df.query(expr=" and ".join(
                        ([" and ".join(
                            map(lambda t_: "`{}` == {}".format(t_, stats_full_agreement["max_{}".format(col)]), t)
                        )] if len(t) >= 1 else []) +
                        ([" and ".join(
                            map(lambda t_: "(`{0}` == -{1} or `{0}`.isna())".format(
                                t_, stats_full_agreement["max_{}".format(col)]
                            ),anti_set)
                        )] if len(anti_set) >= 1 else [])
                    ), inplace=False)
                )

    plot_dict(stats_full_agreement)