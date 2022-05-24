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
#total_samples: Union[int, List[str]] = [
#    "test_1013", "test_1043", "test_1089", "test_1117", "test_1130", "test_1151", "test_1158", "test_1184", "test_1191",
#    "test_1222", "test_635", "test_658", "test_687", "test_743", "test_744", "test_770", "test_775", "test_804",
#    "test_811", "test_832", "test_833", "test_835", "test_861", "test_884", "test_910", "test_915", "test_946",
#    "test_960", "test_978", "test_985"
#]
#total_samples: Union[int, List[str]] = [
#    'test_1012', 'test_1027', 'test_1035', 'test_1068', 'test_1072', 'test_1113', 'test_1137', 'test_1154', 'test_1171',
#    'test_1197', 'test_1203', 'test_1205', 'test_1210', 'test_621', 'test_630', 'test_637', 'test_681', 'test_684',
#    'test_741', 'test_745', 'test_753', 'test_782', 'test_796', 'test_799', 'test_815', 'test_876', 'test_890',
#    'test_911', 'test_930', 'test_942'
#]

# for more annotations
total_samples: Union[int, List[str]] = ['test_1', 'test_2', 'test_3', 'test_6', 'test_11', 'test_15', 'test_16', 'test_18', 'test_26', 'test_36', 'test_40', 'test_42', 'test_45', 'test_47', 'test_52', 'test_53', 'test_56', 'test_58', 'test_64', 'test_65', 'test_66', 'test_70', 'test_76', 'test_79', 'test_85', 'test_92', 'test_93', 'test_94', 'test_97', 'test_98', 'test_103', 'test_104', 'test_106', 'test_107', 'test_108', 'test_110', 'test_112', 'test_113', 'test_114', 'test_119', 'test_120', 'test_121', 'test_123', 'test_128', 'test_139', 'test_145', 'test_148', 'test_149', 'test_172', 'test_175', 'test_184', 'test_186', 'test_188', 'test_190', 'test_192', 'test_196', 'test_197', 'test_199', 'test_203', 'test_215', 'test_216', 'test_217', 'test_226', 'test_230', 'test_239', 'test_247', 'test_248', 'test_251', 'test_259', 'test_262', 'test_264', 'test_273', 'test_276', 'test_285', 'test_303', 'test_307', 'test_312', 'test_314', 'test_317', 'test_318', 'test_319', 'test_324', 'test_331', 'test_335', 'test_337', 'test_338', 'test_342', 'test_343', 'test_353', 'test_364', 'test_365', 'test_374', 'test_377', 'test_382', 'test_383', 'test_384', 'test_387', 'test_391', 'test_402', 'test_412', 'test_419', 'test_421', 'test_422', 'test_423', 'test_426', 'test_442', 'test_444', 'test_447', 'test_449', 'test_451', 'test_455', 'test_460', 'test_470', 'test_480', 'test_488', 'test_491', 'test_492', 'test_496', 'test_499', 'test_505', 'test_509', 'test_512', 'test_514', 'test_515', 'test_517', 'test_519', 'test_523', 'test_524', 'test_525', 'test_527', 'test_532', 'test_533', 'test_539', 'test_542', 'test_545', 'test_548', 'test_553', 'test_555', 'test_556', 'test_557', 'test_558', 'test_578', 'test_582', 'test_584', 'test_590', 'test_592', 'test_596', 'test_597', 'test_598', 'test_599']


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