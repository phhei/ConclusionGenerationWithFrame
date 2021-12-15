from pathlib import Path
import random

import pandas
from typing import List, Union, Tuple
from loguru import logger

from Frames import FrameSet
from MainUtils import retrieve_topic, retrieve_issue_specific_frame, retrieve_generic_frame


def combine_generations(csv: List[Union[Path, str, Tuple[str, Union[Path, str]]]]) -> pandas.DataFrame:
    """
    Combines certain generations in a big (comprehensive) Dataframe-table

    :param csv: a list of cherry-picked predictions
                    <i>(e.g.: cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv)</i>
                    If you want to have an enriched Dataframe, we recommend to use tuples:
                    <tt>(shortcut_string, path), ...</tt>
                    For inferring the used generic frame, at least one shortcut_string must contain <b>generic</b>
                    and for
                    having the issue-specific-frame, at least one string have to contain <b>specific</b>
    :return: An combined enriched dataframe with following special fields (among others):
                <ol>
                    <li>_generic_frame</li>
                    <li>_specific_frame</li>
                    <li>_random_conclusion</li>
                </ol>
    """
    dfs: List[pandas.DataFrame] = []
    for i, t_file in enumerate(csv):
        prefix = "{}_".format(i)
        file = t_file
        if isinstance(t_file, Tuple):
            logger.debug("You specified the index ({}), that's good :)", t_file[0])
            prefix = t_file[0] if t_file[0].endswith("_") else "{}_".format(t_file[0])
            file = t_file[-1]

        logger.info("Let's load \"{}\"", file)
        try:
            dfs.append(pandas.read_csv(filepath_or_buffer=str(file.absolute()) if isinstance(file, Path) else file,
                                       encoding="utf-8",
                                       index_col="test_ID",
                                       verbose=True,
                                       encoding_errors="strict",
                                       on_bad_lines="warn").add_prefix(prefix=prefix))
            logger.success("Successfully load {} lines from \"{}\"", len(dfs[-1]), file)
        except FileNotFoundError:
            logger.opt(exception=True).warning("Please only enter already computed files!")
        except UnicodeError:
            logger.opt(exception=True).error("It's not utf-8 :\\ - skip \"{}\"", file)

    logger.success("Successfully load {} files", len(dfs))

    if len(dfs) >= 2:
        df = dfs[0].join(other=dfs[1:], how="inner")
        logger.success("Joined ({}->{} lines, {} cols)", len(dfs[0]), len(df), len(df.columns))
    elif len(dfs) == 1:
        df = dfs[0]
        logger.info("Easy, nothing to join, only 1 (valid) file was given ({} cols)", len(df.columns))
    else:
        logger.warning("No valid Dataframe was given!")
        return pandas.DataFrame()

    logger.debug("Some things are left to do...")

    topics = []
    generic_frames = []
    specific_frames = []
    random_conclusions = []

    input_rows = [c for c in df.columns if "input" in c]
    generic_frame_cols = [
        c for c in input_rows
        if (not any(map(lambda v: isinstance(v, Tuple), csv)) or
            any(map(lambda v: isinstance(v, Tuple) and "generic" in v[0] and c.startswith(v[0]), csv)))
    ]
    if len(generic_frame_cols) == 0:
        logger.warning("No column contains a cue about the used generic frame - give up...")
        generic_frame_col = None
    else:
        generic_frame_col = generic_frame_cols[0]
    specific_frame_cols = [
        c for c in input_rows
        if (not any(map(lambda v: isinstance(v, Tuple), csv)) or
            any(map(lambda v: isinstance(v, Tuple) and "specific" in v[0] and c.startswith(v[0]), csv)))
    ]
    if len(specific_frame_cols) == 0:
        logger.warning("No column contains a cue about the used issue specific frame - give up...")
        specific_frame_col = None
    else:
        specific_frame_col = specific_frame_cols[0]
    for index, data in df.iterrows():
        logger.trace("Extend row \"{}\" now...", index)

        if len(input_rows) >= 1:
            for i in range(len(input_rows)):
                topic = retrieve_topic(premise=data[input_rows[i]])
                if topic != "not available":
                    logger.trace("Found suitable topic: {}", topic)
                    topics.append(topic)
                    break

        if specific_frame_col is not None:
            specific_frames.append(retrieve_issue_specific_frame(premise=data[specific_frame_col]))
            logger.trace("Added issue-specific frame \"{}\"", specific_frames[-1])
        if generic_frame_col is not None:
            generic_frames.append(retrieve_generic_frame(premise=data[generic_frame_col]))
            logger.trace("Added generic frame \"{}\"", generic_frames[-1])

        random_index = index
        while random_index == index:
            random_index = random.choice(df.index)
        logger.trace("Selects the random index \"{}\" for the current index \"{}\"", random_index, index)
        random_col = random.choice([c for c in df.columns if c.endswith("selected_prediction") or c.endswith("ground_truth")])

        try:
            random_conclusions.append(df[random_col][random_index])
        except KeyError:
            logger.opt(exception=True).error("Index or column not available \"{}\"->\"{}\"", random_index, random_col)
            try:
                random_conclusions.append(df[df.columns[0]][random_index])
            except KeyError:
                logger.opt(exception=True).error("Can't grab anything of row \"{}\" - ERROR VALUE", random_index)
                random_conclusions.append("Malformed random conclusion")

        logger.debug("Extend row \"{}\" with: {}|{}|{}", index,
                     None if specific_frame_col is None else specific_frames[-1],
                     None if generic_frame_col is None else generic_frames[-1],
                     random_conclusions[-1])
    df.insert(loc=0, column="_random_conclusion", value=random_conclusions)
    df.insert(loc=0, column="_specific_frame", value=specific_frames)
    if generic_frame_col is not None:
        df.insert(loc=0, column="_generic_frame", value=generic_frames)
    df.insert(loc=0, column="_topic", value=topics)
    logger.success("Extended the dataframe (now {} cols: {})", len(df), df.columns)

    return df


def translate_generic_frame_readable(df: pandas.DataFrame, frame_set: Union[str, FrameSet]) -> None:
    """
    Using the method ``combine_generations`` only provides the rather cryptic generic frame label. This methods
    converts it into a more readable representation

    :param df: the dataframe to change
    :param frame_set: the frame-set which belongs to. can be directly the loaded frame-set or the plain string
    :return: updates are in-place, hence: nothing to returm
    """
    if "_generic_frame" not in df.columns:
        logger.warning("Nothing to do, column \"_generic_frame\" is missing - apply combine_generations first!")
        return

    if isinstance(frame_set, str):
        logger.debug("We have to precess your raw Frame-set-string first!")
        frame_set: FrameSet = FrameSet(frame_set=frame_set, add_other=True)
        logger.success("Loaded the frame-set: {}", frame_set)
    else:
        logger.info("Frame-set \"{}\" is already loaded", frame_set)

    for i in df.index:
        logger.trace("Processing line } \"{}\"", i)

        df.at[i, "_generic_frame"] = frame_set.issues_specific_frame_to_generic(
            issue_specific_frame=df["_generic_frame"][i],
            fetch_column="label", semantic_reordering=False, remove_stopwords=False
        )

    logger.success("Finished processing {} lines", len(df))
