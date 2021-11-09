from functools import reduce

import pandas
from pathlib import Path
from typing import List, Optional, Dict, Union
from loguru import logger
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from pprint import pprint

from const import AVAILABLE_SCORES


predictions_scores_csv: Path = Path(
    "..",
    ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/128-24/smoothing0.1/tdf0.15/media-frames0.5/t5-large-media-frames/predictions_scores.csv"
)
reference_scores: List[str] = ["rouge1", "rougeL", "bertscore_f1"
                               , "framescore_score"
                               ]
exclude_scores: List[str] = ["rouge", "bertscore_precision", "bertscore_recall", "bertscore_f1"
                             #, "framescore_confidence", "framescore_precision", "framescore_score"
                             ]
save_path: Path = predictions_scores_csv.parent.joinpath("linear_regression_cherry_picker", "-".join(reference_scores))


def average_output(f_predictions_scores_csv: Path = predictions_scores_csv,
                   cherry_picked_files_dir: Optional[Path] = save_path,
                   f_reference_scores: Optional[List[str]] = None,
                   training_samples: Optional[Union[int, float]] = .5) -> Dict[str, float]:
    if f_reference_scores is None:
        f_reference_scores = reference_scores
    df_main = pandas.read_csv(filepath_or_buffer=str(f_predictions_scores_csv.absolute()),
                              encoding="utf-8", index_col="test_ID")
    logger.info("Read {} samples from \"{}\"", len(df_main), f_predictions_scores_csv.name)
    if training_samples is not None:
        if isinstance(training_samples, float):
            df_main = df_main[int(len(df_main)*training_samples)+1:]
        else:
            df_main = df_main[training_samples+1:]
        logger.warning("Some of your test samples were used to train this Cherry-Picker! "
                       "Discard them to prevent spoiling! {} remain", len(df_main))

    f_ret = dict()
    plot_dict = {
        "xticks": [],
        "data": [],
        "mean": []
    }
    try:
        plot_dict["xticks"].extend(["best_beam_{}".format(r) for r in f_reference_scores])
        data = [df_main["best_beam_prediction_{}".format(r)] for r in f_reference_scores]
        mean = [data[r_i].mean(skipna=True) for r_i in range(len(data))]
        f_ret.update(
            {
                "best_beam_avg_{}".format(r): round(mean[d_i], 5)
                for d_i, r in enumerate(f_reference_scores)
            }
        )
        f_ret.update(
            {
                "best_beam_std_{}".format(r): round(data[d_i].std(skipna=True), 5)
                for d_i, r in enumerate(f_reference_scores)
            }
        )
        plot_dict["data"].extend(data)
        plot_dict["mean"].extend(mean)
        plot_dict["xticks"].extend(["worst_{}".format(r) for r in f_reference_scores])
        data = [df_main[[c for c in df.columns if r in c]].min(axis="columns", skipna=True, numeric_only=True)
                for r in f_reference_scores]
        mean = [data[r_i].mean(skipna=True) for r_i in range(len(data))]
        f_ret.update(
            {
                "worst_avg_{}".format(r): round(mean[d_i], 4)
                for d_i, r in enumerate(f_reference_scores)
            }
        )
        plot_dict["data"].extend(data)
        plot_dict["mean"].extend(mean)
        plot_dict["xticks"].extend(["optimal_{}".format(r) for r in f_reference_scores])
        data = [df_main[[c for c in df.columns if r in c]].max(axis="columns", skipna=True, numeric_only=True)
                for r in f_reference_scores]
        mean = [data[r_i].mean(skipna=True) for r_i in range(len(data))]
        f_ret.update(
            {
                "optimal_avg_{}".format(r): round(mean[d_i], 4)
                for d_i, r in enumerate(f_reference_scores)
            }
        )
        f_ret.update(
            {
                "optimal_std_{}".format(r): round(data[d_i].std(skipna=True), 4)
                for d_i, r in enumerate(f_reference_scores)
            }
        )
        plot_dict["data"].extend(data)
        plot_dict["mean"].extend(mean)
    except KeyError:
        logger.opt(exception=True).error("Can't calculated the average beam score. There are two possible reasons: "
                                         "1) malformed prediction_score. Are you sure you already applied the metrics "
                                         "on it? "
                                         "2) Invalid reference scores {}", reference_scores)

    logger.success("Successfully read through \"{}\": {}", f_predictions_scores_csv.name, f_ret)

    if cherry_picked_files_dir is not None:
        for file in cherry_picked_files_dir.glob(pattern="*.csv"):
            logger.info("Found a cherry-picked file: {}", file)
            logger.trace("Try to read \"{}\"", file.absolute())
            df_c = pandas.read_csv(filepath_or_buffer=str(file.absolute()),
                                   encoding="utf-8", index_col="test_ID")
            logger.debug("\"{}\" contains {} samples", file.stem, len(df_c))
            if training_samples is not None:
                if isinstance(training_samples, float):
                    df_c = df_c[int(len(df_c) * training_samples) + 1:]
                else:
                    df_c = df_c[training_samples + 1:]
            try:
                plot_dict["xticks"].extend(
                    ["selected_{}".format(r) for r in f_reference_scores if r in file.parent.name]
                )
                data = [df_c[r] for r in f_reference_scores if r in file.parent.name]
                mean = [data[r_i].mean(skipna=True) for r_i in range(len(data))]
                f_ret["{} --> {}".format(file.parent.name, file.stem)] = {
                    "selected_avg_{}".format(r): round(df_c[r].mean(skipna=True), 4)
                    for r in f_reference_scores if r in file.parent.name
                }
                f_ret["{} --> {}".format(file.parent.name, file.stem)].update({
                    "selected_std_{}".format(r): round(df_c[r].std(skipna=True), 4)
                    for r in f_reference_scores if r in file.parent.name
                })
                plot_dict["data"].extend(data)
                plot_dict["mean"].extend(mean)
                logger.debug("Adding {} stats from \"{}\"",
                             len(f_ret["{} --> {}".format(file.parent.name, file.stem)]), file.stem)
            except KeyError:
                logger.opt(exception=True).warning("\"{}\" is probable no cherry-picked file - ignore...", file.name)

    logger.success("Finished creating a average-stat with {} stats", len(f_ret))

    try:
        with f_predictions_scores_csv.parent.joinpath("avg_stats.txt").open(mode="w", encoding="utf-8") as avg_writer:
            pprint(
                object=f_ret,
                stream=avg_writer,
                sort_dicts=True,
                depth=2,
                indent=4,
                compact=False,
                width=60
            )
    except IOError:
        pprint(object=f_ret, compact=True, sort_dicts=True)

    logger.trace("Return stats: {}", f_ret)

    plt.figure(figsize=(7, 7), dpi=120)
    plt.xticks(range(len(plot_dict["xticks"])), plot_dict["xticks"], rotation=90)
    points = reduce(lambda a, b: a+b, [[(s_i, p) for p in series] for s_i, series in enumerate(plot_dict["data"])])
    plt.scatter([p[0] for p in points], [p[1] for p in points],
                s=80, alpha=0.02)
    plt.plot(range(len(plot_dict["mean"])), plot_dict["mean"], color="b", linewidth=3, alpha=.5)
    plt.grid(b=True, which="major", axis="y", alpha=0.25, linestyle='-', linewidth=2)
    plt.grid(b=True, which="minor", axis="y", alpha=0.15, linestyle='--', linewidth=1)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.show()

    return f_ret


if __name__ == "__main__":
    logger.info("Let's analyse \"{}\"", predictions_scores_csv)
    if not predictions_scores_csv.exists() or not predictions_scores_csv.is_file():
        logger.error("\"{}\" is unknown/ no file! Please give us an existing file!", predictions_scores_csv.absolute())
        exit(-1 if predictions_scores_csv.is_file() else -100)

    df: pandas.DataFrame = pandas.read_csv(
        filepath_or_buffer=str(predictions_scores_csv.absolute()),
        index_col="test_ID",
        encoding="utf-8"
    )

    logger.success("Read {} samples from \"{}\"", len(df), predictions_scores_csv.name)
    logger.debug("Cols: {}", df.columns)

    Y_cols = [col for col in df.columns if any(map(lambda r: r in col, reference_scores))]
    logger.info("Found following columns for y-values: {} (input: {})", Y_cols, ", ".join(reference_scores))
    X_cols = [
        col for col in df.columns
        if any(map(lambda r: r in col, AVAILABLE_SCORES)) and not any(map(lambda r: r in col, exclude_scores))
    ]
    logger.info("Found following columns for x-values: {} (ignore: {})", X_cols, ", ".join(exclude_scores))

    XY_cols_groups = []

    for i in range(128):
        logger.trace("Let's check if we can find a {}. prediction inside the cols {}", X_cols)
        if any(map(lambda c: "_{}_".format(i) in c, X_cols)):
            logger.debug("Found a {}. sample in the line!", i)
            X_cols_i = [col for col in X_cols if "_{}_".format(i) in col]
            Y_cols_i = [col for col in Y_cols if "_{}_".format(i) in col]
            if len(Y_cols_i) == len(reference_scores):
                logger.success("Found the {} references scores in the {}. sample: {}", len(reference_scores), i,
                               Y_cols_i)
            else:
                logger.warning("Expected {} references cols, but found {}! {}", len(reference_scores), len(Y_cols_i),
                               Y_cols_i)
            logger.debug("Aligned with: {}", X_cols_i)
            XY_cols_groups.append((X_cols_i, Y_cols_i))

            for col in X_cols_i:
                X_cols.remove(col)
                logger.trace("Removed column \"{}\" from the X_cols-list ({} left)", col, len(X_cols))
            for col in Y_cols_i:
                Y_cols.remove(col)
                logger.trace("Removed column \"{}\" from the Y_cols-list ({} left)", col, len(Y_cols))
            logger.success("Processed the cols for the {}. prediction. {}#{} entries left", i, len(X_cols), len(Y_cols))
        else:
            logger.debug("No more samples - max {}", i-1)
            break

    if len(X_cols) >= 1 and len(Y_cols) >= 1:
        logger.info("There are some cols left: {}#{}", X_cols, Y_cols)
        XY_cols_groups.append((X_cols, Y_cols))

    logger.success("Finished grouping - ended up with {} groups", len(XY_cols_groups))
    try:
        logger.info("A group is structured as follows: {} -> {}", XY_cols_groups[0][0], XY_cols_groups[0][1])
    except IndexError:
        logger.opt(exception=True).critical("Your CSV \"{}\" is malformed!", )

    X = []
    Y = []

    for i, row in df.iterrows():
        logger.debug("Let's iterate over the row \"{}\"", i)
        for X_current_cols, Y_current_cols in XY_cols_groups:
            try:
                X.append([float(row[x]) for x in X_current_cols])
                Y.append(sum([float(row[y]) for y in Y_current_cols])/len(Y_current_cols))
                logger.trace("Appended {} -> {}", X[-1], Y[-1])
                if Y[-1] >= .95:
                    logger.info("Sample \"{}\" is a proper sample :) [\"{}\" <-> \"{}\"]",
                                i, row.get("best_beam_prediction", "n/a"), row.get("ground_truth", "n/a"))
            except KeyError:
                logger.opt(exception=True).critical("PANDAS!")
            except ValueError:
                logger.opt(exception=True).warning("Row \"{}\" :: \"{}\" is strange...", i, row)
        logger.debug("Collected training samples in row \"{}\"", i)
    logger.success("Successfully loaded {} training samples!", len(X))

    model = LinearRegression(fit_intercept=False, copy_X=True, positive=False)
    X_train = X[:int(len(X)*.5)]
    X_train, norms = normalize(X=X_train, norm="max", axis=0, copy=True, return_norm=True)
    logger.info("We normalized the training data with following norms: {}", norms)
    X_test = [[feature/norm for feature, norm in zip(sample, norms)] for sample in X[int(len(X)*.5):]]
    Y_train = Y[:int(len(Y)*.5)]
    Y_test = Y[int(len(Y)*.5):]
    logger.info("Initializes a linear model {}, will train with {} samples (out of {})", model, len(X_train), len(X))

    logger.success("Trained: {}", model.fit(X=X_train, y=Y_train))
    final_score = model.score(X=X_train, y=Y_train)
    model_dict = {"final_score": final_score}
    for i, param in enumerate(model.coef_):
        try:
            model_dict[XY_cols_groups[0][0][i]] = param
        except IndexError:
            logger.opt(exception=True).info("Pending param {} -> {}", i, param)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        with save_path.joinpath("stats.txt").open(mode="w", encoding="utf-8") as dump:
            pprint(
                object=model_dict,
                stream=dump,
                sort_dicts=True,
                indent=1,
                compact=False,
                width=100
            )
            logger.debug("Saved stats ({}) to \"{}\"", len(model_dict), dump.name)
    except IOError:
        pprint(object=model_dict, compact=True)

    logger.info("OK, let's cherry-pick with \"{}\"", model)
    ret = dict()
    for i, row in df.iterrows():
        X_prediction = []
        for X_current_cols, _ in XY_cols_groups:
            X_prediction.append([float(row[x])/norms[ci] for ci, x in enumerate(X_current_cols)])
            logger.trace("Appended {} -> {}", X[-1])
        logger.trace("Decide between {} samples...", len(X_prediction))
        scores = [(j, s) for j, s in enumerate(model.predict(X=X_prediction))]
        scores.sort(key=lambda st: st[1], reverse=True)
        logger.debug("Sample \"{}\": {}", i, " > ".join(map(lambda s: "{}: {}".format(s[0], round(s[1], 3)), scores)))
        best_j = scores[0][0]
        best_key:str = \
            [key[:-len(reference_scores[0])] for key in XY_cols_groups[best_j][1] if reference_scores[0] in key][0].\
                rstrip(" _")
        logger.info("For the sample \"{}\", the best prediction is the {}. one: {}", i, best_j, best_key)
        try:
            ret[i] = {
                "input": row["input"],
                "ground_truth": row["ground_truth"],
                "selected_prediction": row[best_key]
            }
            ret[i]["selected_prediction"] = row[best_key]
            ret[i]["selected_prediction_pos"] = best_j
            ret[i].update({[r for r in reference_scores if r in key][0]: row[key] for key in XY_cols_groups[best_j][1]})
        except KeyError:
            logger.opt(exception=True).error("Malformed input...")
    logger.success("Got {} predictions/ lines", len(ret))
    try:
        csv_path: Path = save_path.joinpath("cherry_picked_without-{}.csv".format("".join(exclude_scores)))
        pandas.DataFrame.from_dict(data=ret, orient="index").to_csv(
            path_or_buf=str(csv_path.absolute()),
            encoding="utf-8",
            index_label="test_ID",
            errors="ignore"
        )
        logger.success("Saved the cherry-picked conclusions to \"{}\"", csv_path)
    except IOError:
        logger.opt(exception=True).warning("Cannot write the cherry-picked conclusions to a CSV :\\")
        pprint(
            object=ret,
            indent=2,
            width=200,
            depth=2,
            compact=False,
            sort_dicts=True
        )

    average_output()
