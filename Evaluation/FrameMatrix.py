import os

import pandas
from typing import List, Optional

from Frames import FrameSet
from pathlib import Path
from loguru import logger

from MainUtils import retrieve_generic_frame

os.chdir("../")

anchor_path: Optional[str] = ""
original_framed_cherry_picked: str = ""
other_framed_cherry_picked: List[str] = []

original_frame_set: Optional[FrameSet] = FrameSet(frame_set="media_frames", add_other=True)
alternative_frame_set: Optional[FrameSet] = FrameSet(frame_set="frequent_media_frames", add_other=False)

include_original_frame: bool = True
include_other: bool = True
exclude_unfitting_original_samples: bool = True
exclude_unfitting_modified_samples: bool = True

columns_for_rating = ["bertscore_f1", "rouge1", "rougeL"]


def extract_modified_frame(path: str) -> str:
    folder_name = Path(path).parent.parent.parent.name
    if not folder_name.startswith("predictions_scores_"):
        return folder_name

    folder_name = folder_name[len("predictions_scores_"):]
    if folder_name.startswith(tuple(str(i) for i in range(1, 10))):
        folder_name = folder_name.split(sep="_", maxsplit=1)[1]
    if folder_name.endswith("_linear_regression_cherry_picker"):
        folder_name = folder_name[:-len("_linear_regression_cherry_picker")]
    if folder_name.endswith("_cherry_picker"):
        folder_name = folder_name[:-len("_cherry_picker")]

    return folder_name


if __name__ == "__main__":
    try:
        df_original_framed_cherry_picked = pandas.read_csv(filepath_or_buffer=original_framed_cherry_picked,
                                                           encoding="utf-8", encoding_errors="ignore",
                                                           index_col="test_ID")
        logger.success("Reads successfully the original framed CSV: {} lines", len(df_original_framed_cherry_picked))
        df_other_framed_cherry_picked = \
            [(pandas.read_csv(filepath_or_buffer=o, encoding="utf-8", encoding_errors="ignore", index_col="test_ID"),
              extract_modified_frame(path=o))
             for o in other_framed_cherry_picked]
        logger.success("And all the {} alternative framed CSVs: {}", len(df_other_framed_cherry_picked),
                       ", ".join(map(lambda d: d[1], df_other_framed_cherry_picked)))
    except FileNotFoundError:
        logger.opt(exception=True).critical("File not found - please provide the whole path including the cherry_picked"
                                            "....csv-file")
        exit(-1)

    if alternative_frame_set is None:
        # noinspection PyUnboundLocalVariable
        alternative_frames = {lbl for df, lbl in df_other_framed_cherry_picked}
    else:
        alternative_frames = set(alternative_frame_set.data["label"].values)
    logger.info("OK, we'll test all original frames: {} ({})", len(alternative_frames), alternative_frame_set)

    if include_original_frame:
        alternative_frames.add("original")
    if include_other:
        alternative_frames.add("other")
        logger.trace("Add other -> {}", len(alternative_frames))

    try:
        for df, _ in df_other_framed_cherry_picked:
            if alternative_frame_set is not None and exclude_unfitting_modified_samples:
                df.drop(
                    index=[i for i, c in df.iterrows()
                           if retrieve_generic_frame(premise=c["input"]).lower()
                           not in alternative_frame_set.data["keywords_label"].values],
                    inplace=True
                )
            df.drop(
                columns=set(df.columns).difference(
                    ["selected_prediction", "selected_prediction_pos"] + columns_for_rating
                ),
                inplace=True
            )

        df_other_framed_cherry_picked = [df.add_prefix("{}#".format(prefix))
                                         for df, prefix in df_other_framed_cherry_picked if not df.empty]

        if original_frame_set is not None and exclude_unfitting_original_samples:
            valid_frame_labels = set(original_frame_set.data["keywords_label"].values)
            if include_other:
                valid_frame_labels.add("other")
            # noinspection PyUnboundLocalVariable
            df_original_framed_cherry_picked.drop(
                index=[i for i, c in df_original_framed_cherry_picked.iterrows()
                       if retrieve_generic_frame(premise=c["input"], default="other").lower() not in valid_frame_labels],
                inplace=True
            )

            if df_original_framed_cherry_picked.empty:
                logger.warning("Your original framed set is empty now, because no frame fits to {}", original_frame_set)
                exit(1)
    except KeyError:
        logger.opt(exception=True).error("Your CSV-files need to have the column \"input\"")
        exit(-hash("input"))

    if original_frame_set is None:
        original_frames = \
            {retrieve_generic_frame(premise=d["input"]) for _, d in df_original_framed_cherry_picked.iterrows()}
    else:
        original_frames = set(original_frame_set.data["label"].values)
    logger.info("OK, we'll test all original frames: {} ({})", len(original_frames), original_frame_set)

    if include_other:
        original_frames.add("other")
        logger.trace("Add other -> {}", len(original_frames))

    df_original_framed_cherry_picked.drop(
        columns=set(df_original_framed_cherry_picked.columns).difference(
                    ["input", "ground_truth", "selected_prediction", "selected_prediction_pos"] + columns_for_rating
                ),
        inplace=True
    )
    df_original_framed_cherry_picked = df_original_framed_cherry_picked.add_prefix("original#")

    big_df = df_original_framed_cherry_picked.join(
        other=df_other_framed_cherry_picked,
        how="left",
        sort=True
    )
    logger.success("Finally receives a big joined dataframe of {} rows", len(big_df))
    logger.debug("Columns: {}", big_df.columns)

    logger.trace("OK, let's move on to the final matrix!")

    matrix = dict()
    for original in original_frames:
        logger.info("#########################Analyse the \"{}\"-original framed samples#########################",
                    original)
        matrix[original] = dict()
        important_rows: pandas.DataFrame = big_df.drop(
            index=[i for i, d in big_df.iterrows() if
                   (original_frame_set is None and retrieve_generic_frame(d["original#input"]) != original) or
                   (original_frame_set is not None and original_frame_set.issues_specific_frame_to_generic(
                       issue_specific_frame=retrieve_generic_frame(d["original#input"]),
                       semantic_reordering=False,
                       fetch_column="label") != original)],
            inplace=False
        )
        logger.debug("{} samples are \"{}\"-originally framed", len(important_rows), original)
        for alternative in alternative_frames:
            logger.debug("Deeper analyses \"{}\"->\"{}\"", original, alternative)
            matrix[original][alternative] = dict()
            important_cols = \
                {c for c in important_rows.columns
                 if (alternative == "original" and c.startswith("original#"))
                 or (alternative != "original" and not c.startswith("original#") and
                     ((alternative_frame_set is None and c.startswith("{}#".format(columns_for_rating)))
                      or (alternative_frame_set is not None and
                          alternative_frame_set.issues_specific_frame_to_generic(
                              issue_specific_frame=c.split("#", maxsplit=1)[0].replace("-", " "),
                              semantic_reordering=False,
                              fetch_column="label"
                          ).lower() == alternative.replace("-", " ").replace("_", " ").lower())))}
            logger.debug("{} columns are relevant for the alternative farming of \"{}\": {}", len(important_cols),
                         alternative, ", ".join(important_cols))
            for rating in columns_for_rating:
                important_cols_rating = {c for c in important_rows.columns if c in important_cols and rating in c}
                if len(important_cols_rating) == 0:
                    logger.warning("Cell \"{}\"->\"{}\"->\"{}\"... not in database ({})", original, alternative, rating,
                                   important_rows.columns)
                    matrix[original][alternative][rating] = {
                        "mean": -1,
                        "max": -1,
                        "min": -1,
                        "std": -1,
                        "proc_better_alternative": 0
                    }
                else:
                    if len(important_cols_rating) > 1:
                        logger.warning("Cell \"{}\"->\"{}\"->\"{}\"... too many ({}) in database ({})", original,
                                       alternative, rating, len(important_cols_rating), important_rows.columns)
                    fixed_col = important_cols_rating.pop()
                    try:
                        better_alternative = \
                            important_rows.fillna(value=-1, inplace=False)[important_rows["original#{}".format(rating)] <
                                                                           important_rows[fixed_col]]
                    except KeyError:
                        logger.opt(exception=True).info("We're not able to count the rows in which the alternative "
                                                        "frame \"{}\" leads to better results (=more similar to the "
                                                        "ground truth framed by \"{}\")", alternative, original)
                        better_alternative = pandas.DataFrame(data=[], columns=important_rows.columns)
                    matrix[original][alternative][rating] = {
                        "mean": important_rows[fixed_col].mean(skipna=True),
                        "max": important_rows[fixed_col].max(skipna=True),
                        "min": important_rows[fixed_col].min(skipna=True),
                        "std": important_rows[fixed_col].std(skipna=True),
                        "proc_better_alternative":
                            round(100.*len(better_alternative)/len(important_rows), 1)
                            if len(important_rows) >= 1 else 0
                    }
                    if alternative == "original":
                        logger.debug("By random, {}% of samples outperform its own generation with respect to \"{}\"",
                                     matrix[original][alternative][rating]["proc_better_alternative"], rating)
                    else:
                        for rid, data in better_alternative.iterrows():
                            logger.info("The sample \"{}\" reached a better score in \"{}\" "
                                        "framed by the alternative \"{}\"", rid, rating, alternative)
                            better_conclusion_col = {c for c in important_cols if c.endswith("selected_prediction")}
                            logger.debug("Sample \"{}\": \"{}\" -> \"{}\" (better than \"{}\" w.r.t.\"{}\")",
                                         rid,
                                         data.get("original#input", "n/a"),
                                         data.get(better_conclusion_col.pop(), "n/a")
                                         if len(better_conclusion_col) >= 1 else "nothing to show",
                                         data.get("original#selected_prediction", "n/a"),
                                         data.get("original#ground_truth", "n/a"))

    df_matrix = pandas.DataFrame.from_dict(data=matrix, orient="index")
    df_matrix.style.format(precision=3, na_rep="N/A", thousands=" ", decimal=".")
    df_matrix.style.highlight_max(axis=0, color="green")
    df_matrix.style.highlight_min(axis=0, color="gray")
    df_matrix.style.highlight_max(axis=1, props="font-weight: bold;")
    df_matrix.style.highlight_min(axis=1, props="font-weight: lighter;")
    df_matrix.style.background_gradient(low=0, high=1)

    logger.success("Finished: {}", df_matrix.to_string(max_colwidth=15))

    if anchor_path is not None:
        Path(anchor_path).joinpath("frame_matrix.html").write_text(data=df_matrix.to_html(notebook=False, border=1),
                                                                   encoding="utf-8", errors="ignore")
