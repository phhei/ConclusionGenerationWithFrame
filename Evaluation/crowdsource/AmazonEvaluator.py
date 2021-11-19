#  Copyright (c) 2021. Written by Philipp Heinisch for research purposes

import pathlib
import loguru
import numpy
import pandas
import matplotlib.pyplot as plt
import pprint

from typing import Optional, List

anchor_path = pathlib.Path("..", "..", ".out", "manual_evaluation", "survey", "AmazonTurk-0", "results")
mode = None
index_mapping = {"_random_conclusion": "random",
                 "no_frame_0104_ground_truth": "original",
                 "no_frame_0104_selected_prediction": "T5",
                 "specific_0104_selected_prediction": "T5frame",
                 "generic_010405_selected_prediction": "T5gen",
                 "generic_010405_framescore_selected_prediction": "T5genSel"
                 }
max_number_worker_per_task = 5

logger = loguru.logger
pp = pprint.PrettyPrinter(indent=2, width=120, depth=3, sort_dicts=True)


def stat(do_restrictive_round: bool = True, excluded_users: Optional[List[str]] = None):
    distribution = dict()
    aspect_argument_pairwise_matrix = dict()
    worker_random_fails = dict()
    worker_participation = dict()

    logger.info("Let's crawl through \"{}\", collecting the mode \"{}\"", anchor_path.absolute(), mode)

    for folder in ((anchor_path,) if mode is None else anchor_path.glob(pattern="{}*".format(mode))):
        logger.trace("Found folder/ file in \"{}\": \"{}\"", anchor_path.absolute(), folder.name)
        if folder.is_file():
            logger.info("\"{}\" is a file, but we're looking for folders!", folder.name)
        else:
            logger.debug("OK, we dive into \"{}\"", folder.name)
            for result in folder.glob(pattern="*.csv"):
                logger.info("This file seems to contain interesting survey details: {}->{}",
                            result.parent.name, result.name)
                result_df = pandas.read_csv(str(result.absolute()))
                logger.debug("Read {} lines: {}", len(result_df), result_df.columns)
                for line in range(len(result_df)):
                    logger.trace("Read {}. line", line)

                    try:
                        if result_df["AssignmentStatus"][line] == "Rejected" or \
                                (excluded_users is not None and result_df["WorkerId"][line] in excluded_users):
                            logger.info("Worker {} is rejected/ banished in \"{}\", ignore!",
                                        result_df["WorkerId"][line], result.parent.name)
                        else:
                            if result_df["WorkTimeInSeconds"][line] < 300:
                                logger.warning("Worker {} rushes in {}min over the survey",
                                               result_df["WorkerId"][line],
                                               round(result_df["WorkTimeInSeconds"][line] / 60., 2))
                            elif result_df["WorkTimeInSeconds"][line] < 720:
                                logger.info("Worker {} rushes in {}min over the survey",
                                            result_df["WorkerId"][line],
                                            round(result_df["WorkTimeInSeconds"][line] / 60., 1))
                            worker_participation[result_df["WorkerId"][line]] = \
                                worker_participation.get(result_df["WorkerId"][line], set()).union({result.name})
                            json = pandas.read_json(result_df["Answer.taskAnswers"][line])
                            logger.info("Reading the answers now: {}", json.columns)
                            for name, value in json.items():
                                try:
                                    name_split = name.split("_")
                                    argument_id = "_".join(name_split[:2])
                                    aspect = name_split[2]
                                    votes = [v for k, v in index_mapping.items() if k in name]
                                    votes_encrypt = [k for k, v in index_mapping.items() if k in name]
                                    votes.sort()
                                    votes_encrypt.sort(key=lambda v: index_mapping[v])
                                    if len(votes) == 2:
                                        logger.trace("Found the contents: {}", " <-> ".join(votes))

                                        value = value[0]

                                        distribution[aspect] = distribution.get(aspect, dict())
                                        aspect_argument_pairwise_matrix[aspect] = aspect_argument_pairwise_matrix.get(
                                            aspect, dict())
                                        aspect_argument_pairwise_matrix[aspect][argument_id] = \
                                            aspect_argument_pairwise_matrix[aspect].get(argument_id, dict())
                                        fight_key = " <-> ".join(votes)

                                        if value.get("conclusion1", False):
                                            counted_index = int(name.index(votes_encrypt[0]) > name.index(votes_encrypt[1]))
                                        elif value.get("conclusion2", False):
                                            counted_index = int(name.index(votes_encrypt[0]) < name.index(votes_encrypt[1]))
                                        else:
                                            logger.error("No resolvable value in \"{}\": {} ({} -> Worker {})", name,
                                                         value,
                                                         result_df["HITId"][line], result_df["WorkerId"][line])
                                            counted_index = -1
                                        opposite_index = 1 if counted_index == 0 else 0

                                        if counted_index >= 0:
                                            distribution[aspect][votes[counted_index]] = \
                                                distribution[aspect].get(votes[counted_index], dict())
                                            distribution[aspect][votes[counted_index]][argument_id] = \
                                                distribution[aspect][votes[counted_index]].get(argument_id, list())
                                            distribution[aspect][votes[counted_index]][argument_id] += [
                                                result_df["WorkerId"][line]]

                                            aspect_argument_pairwise_matrix[aspect][argument_id][fight_key] = \
                                                aspect_argument_pairwise_matrix[aspect][argument_id].get(fight_key, 0) + \
                                                (-1 if counted_index == 0 else 1)

                                            if votes[counted_index] == "random" and aspect == "Validity":
                                                logger.debug("Worker \"{}\" voted for the random conclusion (mode: {})"
                                                             " -- left",
                                                             result_df["WorkerId"][line], result.parent.name)
                                                worker_random_fails[result_df["WorkerId"][line]] = \
                                                    worker_random_fails.get(result_df["WorkerId"][line],
                                                                            numpy.zeros((2,), dtype=numpy.int)) + \
                                                    (3 if votes[opposite_index] == "original" else 1)
                                            elif votes[opposite_index] == "random" and aspect == "Validity":
                                                worker_random_fails[result_df["WorkerId"][line]] = \
                                                    worker_random_fails.get(result_df["WorkerId"][line],
                                                                            numpy.zeros((2,), dtype=numpy.int))
                                                worker_random_fails[result_df["WorkerId"][line]][1] += \
                                                    3 if votes[counted_index] == "original" else 1
                                    else:
                                        logger.warning("Malformed CSV-field/ non-fitting index_mapping! {} items of {} "
                                                       "found (2 required) in \"{}\"", len(votes), index_mapping.keys(),
                                                       name)
                                except IndexError:
                                    logger.opt(exception=False).error("Corrupted json field: {}", name)
                    except KeyError:
                        logger.opt(exception=True).warning("This file ({}) is a corrupted one!", result.absolute())

    for worker, fails in worker_random_fails.items():
        fails_percentage = float(fails[0]) / float(fails[1])

        if fails_percentage >= .5:
            logger.warning("Worker {} may misunderstood the task ({}) -- random-Validity-failure-rate of {}%",
                           worker, ", ".join(worker_participation.get(worker, set())), round(fails_percentage * 100.))
        elif fails_percentage > .2:
            logger.info("Worker {} may rushed through the task ({}) without looking -- "
                        "random-Validity-failure-rate of {}%",
                        worker, ", ".join(worker_participation.get(worker, set())), round(fails_percentage * 100.))
        else:
            logger.debug("Worker {} had a random-Validity-failure-rate of {}%", worker, round(fails_percentage * 100.))

    aspect_pairwise_matrix = dict()
    for aspect, v in aspect_argument_pairwise_matrix.items():
        for argument, vv in v.items():
            for fight, voting in vv.items():
                if voting > 0:
                    aspect_pairwise_matrix[aspect] = aspect_pairwise_matrix.get(aspect, dict())
                    aspect_pairwise_matrix[aspect][fight.replace("<->", "<")] = \
                        round(aspect_pairwise_matrix[aspect].get(fight.replace("<->", "<"), 0.) +
                              1. + (0.01 if voting >= max_number_worker_per_task / 3 else 0), 2)
                elif voting < 0:
                    aspect_pairwise_matrix[aspect] = aspect_pairwise_matrix.get(aspect, dict())
                    key_aspect = fight.split(" <-> ")
                    key_aspect = "{} < {}".format(key_aspect[-1], key_aspect[0])
                    aspect_pairwise_matrix[aspect][key_aspect] = \
                        round(aspect_pairwise_matrix[aspect].get(key_aspect, 0.) +
                              1. + (0.01 if voting <= -max_number_worker_per_task / 3 else 0), 2)

    logger.success("Created a distribution and the fight_matrix:")
    logger.success(pp.pformat(aspect_pairwise_matrix))

    logger.debug("To be specific, aspect_argument_pairwise_matrix: {}", pp.pformat(aspect_argument_pairwise_matrix))

    logger.info("General votes:")
    general_votes = {k: {ik: sum(map(lambda iiv: len(iiv), iv.values())) for ik, iv in v.items()}
                     for k, v in distribution.items()}
    logger.info(pp.pformat(general_votes))

    # control
    # print(aspect_pairwise_matrix)
    sorted_categories = list(index_mapping.values())
    sorted_categories.sort()
    empty_table = [["" for c in range(len(index_mapping))] for r in range(len(index_mapping))]

    # adds the data
    for key_aspect, value in aspect_pairwise_matrix.items():
        col_data = ["{}\n{}".format(
            c, "".join(["✪"] * sum(map(lambda m: general_votes[key_aspect].get(c, -1) > m,
                                       general_votes[key_aspect].values()))))
            for c in sorted_categories]
        for key2, value2 in value.items():
            # searches for position in index_distribution
            for element_index, element in enumerate(sorted_categories):
                if key2.split()[0] == element:
                    # enters the value to the table
                    total_samples = len(aspect_argument_pairwise_matrix.get(key_aspect))
                    x = float(value2//1)
                    y = round((value2-x)*100*100/total_samples)
                    x = round((x*100.)/total_samples)
                    empty_table[element_index][sorted_categories.index(key2.split(" < ")[1])] = "{}%\n ({}%)".format(x, y)
        fig, ax = plt.subplots(figsize=[3.75, 3])
        ax.set_axis_off()
        table = ax.table(
            cellText=empty_table,
            rowLabels=sorted_categories,
            colLabels=col_data,
            # colours if needed
            rowColours=["cornflowerblue"] * len(sorted_categories),
            colColours=["cornflowerblue"] * len(sorted_categories),
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        cmap = plt.cm.get_cmap("Blues")
        for (row, col), cell in table.get_celld().items():
            cell.set_height(.25)
            cell.set_width(.25)
            if row-1 == col:
                cell.set_facecolor("gray")
            elif row-1 >= 0 and col >= 0:
                try:
                    cell.set_facecolor(
                        cmap(float(cell.get_text().get_text()[:cell.get_text().get_text().index("%")])/100., alpha=0.5)
                    )
                except ValueError:
                    cell.set_facecolor(
                        cmap(0., alpha=0.5)
                    )
                cell.set_fontsize(cell.get_fontsize() + 1.5)
        plt.text(-0.46 if len(sorted_categories) >= 5 else -0.14, 1.275 if len(sorted_categories) >= 5 else 1.025,
                 "<", horizontalalignment='left', verticalalignment='top')
        #table.auto_set_column_width(col=range(1, len(index_distribution)))
        plt.subplots_adjust(left=0.3)
        # title
        #ax.set_title('{}_{}'.format(mode, key), fontweight="bold")
        # saves table
        plot_path = anchor_path.joinpath("tables")
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path.joinpath("{}{}{}.pdf".format(mode, "_" if excluded_users is None else "_restrictive_",
                                                           key_aspect)), bbox_inches="tight")

        if do_restrictive_round:
            stat(do_restrictive_round=False,
                 excluded_users=[wid for wid, rat in worker_random_fails.items()
                                 if float(rat[0]) / float(rat[1]) >= .25])


if __name__ == "__main__":
    stat()