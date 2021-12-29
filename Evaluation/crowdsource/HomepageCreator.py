import itertools
import os
import random
from typing import Union, List, Iterable

from Evaluation.crowdsource.Utils import combine_generations, translate_generic_frame_readable
from FrameClassifier import FrameSet
from MainUtils import clean_premise

os.chdir("../../")

annotation_round = 1
csv = [
    ("T5", ".out/pytorch_lightning/T5ForConditionalGeneration/128-24/t5-large-without-frame-and-topic/predictions.csv"),
    ("T5+topic", ".out/pytorch_lightning/T5ForConditionalGeneration/132-24/t5-large-only-topic/predictions.csv"),
    ("T5+topic+is", ".out/pytorch_lightning/T5ForConditionalGeneration/128-24/t5-large-res/predictions.csv"),
    ("T5+topic+gen", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/128-24/t5-large-media-frames/predictions.csv"),
    ("T5+topic+inf", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/136-24/t5-large-media-frames-topic-inf-frame/predictions.csv"),
    ("T5+topic+specific+generic", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/140-24/t5-large-media-frames-topic-is-gen-frame/predictions.csv"),
    ("T5+topic+specific+inf", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/140-24/t5-large-media-frames-topic-is-inf-frame/predictions.csv"),
    ("T5+topic+specific+generic+inf", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/144-24/lightning_logs/T5-large-media_frames-is_gen_inf/predictions.csv")
]
column_for_premise = "T5+topic_input"
columns_for_conclusion = ["_random_conclusion",
                          "T5_ground_truth", "T5_best_beam_prediction",
                          "T5+topic_best_beam_prediction",
                          "T5+topic+is_best_beam_prediction",
                          "T5+topic+gen_best_beam_prediction",
                          "T5+topic+inf_best_beam_prediction",
                          "T5+topic+specific+generic_best_beam_prediction",
                          "T5+topic+specific+inf_best_beam_prediction",
                          "T5+topic+specific+generic+inf_best_beam_prediction"]
skip_first_half: bool = True
skip_other_framed: bool = False
split_conclusion_batches: Union[bool, int] = True
skip_generic_frame_question_when_other: bool = True
skip_equal_matches_number: int = 5
total_samples: Union[int, List[str]] = 30

frame_set = FrameSet(frame_set="media_frames", add_other=True)
#frame_set.add_ecology_frame()


if __name__ == "__main__":
    same_conclusion_count = dict()

    df = combine_generations(csv)
    translate_generic_frame_readable(df=df, frame_set=frame_set)

    if skip_first_half:
        df = df[int(len(df)*.5):]

    if skip_other_framed:
        df = df[df._generic_frame != "other"]

    if skip_equal_matches_number >= 2:
        cols_perm = itertools.combinations(
            columns_for_conclusion if isinstance(columns_for_conclusion[0], str)
            else itertools.chain(*columns_for_conclusion),
            skip_equal_matches_number
        )
        for perm in cols_perm:
            query_string = " == ".join(map(lambda p: "`{}`".format(p), perm))
            df = df.drop(index=df.query(query_string).index, errors="ignore")

    map_increment_to_index = {i: e for i, e in enumerate(df.index)}

    sql_string_arguments = "INSERT IGNORE INTO CrowdSourceArgument " \
                           "(argument_ID, topic, premise, issue_specific_frame, generic_mapped_frame) VALUES "
    sql_string_conclusions = "INSERT INTO CrowdSourceConclusion(argument_ID, conclusion_identifier, conclusion_text, " \
                             "order_number, round) VALUES "

    sql_list_arguments = []
    sql_list_conclusions = []
    iter_index = total_samples if isinstance(total_samples, Iterable) else \
        df.sample(n=total_samples, replace=False, axis="index",
                  random_state=30, ignore_index=False).index
    for index in iter_index:
        sql_list_arguments.append(
            "(\"{}\", \"{}\", \"{}\", \"{}\", \"{}\")".format(
                index,
                df["_topic"][index],
                clean_premise(df[column_for_premise][index]).replace("\"", "&quot;"),
                df["_specific_frame"][index].replace("\"", "&quot;"),
                "NULL" if skip_generic_frame_question_when_other and df["_generic_frame"][index] == "other" else
                df["_generic_frame"][index].replace("\"", "&quot;")
            )
        )
        random.shuffle(columns_for_conclusion)
        for i, conclusion_column in enumerate(columns_for_conclusion):
            sql_list_conclusions.append(
                "(\"{0}\", \"{1}\", \"{2}\", {3}, {4})".format(
                    index,
                    conclusion_column,
                    df[conclusion_column][index].replace("\"", "&quot;"),
                    (i % split_conclusion_batches if type(split_conclusion_batches).__name__ == "int" else i % 2)
                    if (type(split_conclusion_batches).__name__ == "int" or split_conclusion_batches) else 0,
                    annotation_round
                )
            )

    sql_string_arguments += ", ".join(sql_list_arguments)
    sql_string_arguments += ";"
    sql_string_conclusions += ", ".join(sql_list_conclusions)
    sql_string_conclusions += " ON DUPLICATE KEY UPDATE round={};".format(annotation_round)

    print(sql_string_arguments)
    print(sql_string_conclusions)