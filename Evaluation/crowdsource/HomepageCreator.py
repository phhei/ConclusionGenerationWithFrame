import itertools
import os
import pprint
import random

from Evaluation.crowdsource.Utils import combine_generations, translate_generic_frame_readable
from FrameClassifier import FrameSet
from MainUtils import clean_premise

os.chdir("../../")

csv = [
    ("no_frame_0104", ".out/pytorch_lightning/T5ForConditionalGeneration/128-24/smoothing0.1/idf0.4/t5-large-without-frame/linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv"),
    ("specific_0104", ".out/pytorch_lightning/T5ForConditionalGeneration/128-24/smoothing0.1/idf0.4/t5-large-res/linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv"),
    ("generic_010405", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/128-24/smoothing0.1/tdf0.4/media-frames0.5/t5-large-media-frames/predictions_scores_linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv"),
    ("generic_010405_framescore", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/128-24/smoothing0.1/tdf0.4/media-frames0.5/t5-large-media-frames/predictions_scores_linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1-framescore_score/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv")
]
column_for_premise = "specific_0104_input"
columns_for_comparison = ["_random_conclusion",
                          "no_frame_0104_ground_truth", "no_frame_0104_selected_prediction",
                          "specific_0104_selected_prediction",
                          "generic_010405_selected_prediction",
                          "generic_010405_framescore_selected_prediction"]
skip_first_half: bool = True
skip_other: bool = False
skip_question_if_conclusions_are_equal: bool = True
skip_generic_frame_question_when_other: bool = True
skip_equal_matches_number: int = 4
total_samples = 10

frame_set = FrameSet(frame_set="media_frames", add_other=True)
#frame_set.add_ecology_frame()


if __name__ == "__main__":
    same_conclusion_count = dict()

    df = combine_generations(csv)
    translate_generic_frame_readable(df=df, frame_set=frame_set)

    if skip_first_half:
        df = df[int(len(df)*.5):]

    if skip_other:
        df = df[df._generic_frame != "other"]

    if skip_equal_matches_number >= 2:
        cols_perm = itertools.combinations(columns_for_comparison, skip_equal_matches_number)
        for perm in cols_perm:
            query_string = " == ".join(map(lambda p: "`{}`".format(p), perm))
            df = df.drop(index=df.query(query_string).index, errors="ignore")

    map_increment_to_index = {i: e for i, e in enumerate(df.index)}

    sql_string = "INSERT INTO CrowdSourceSamples (topic, premise, conclusion1ID, conclusion1, conclusion2ID, " \
                 "conclusion2, issuespecificframe, genericmappedframe, genericinferredframe) VALUES "

    sql_strings = []
    scalar = int(len(df) / total_samples)
    for survey_id in range(0, total_samples):
        index = map_increment_to_index[survey_id]
        combos = [random.choice([(left, right), (right, left)])
                  for left, right in itertools.combinations(columns_for_comparison, 2)]
        random.shuffle(combos)
        for i, combo in enumerate(combos):
            same_conclusion_count["total"] = same_conclusion_count.get("total", 0) + 1
            if skip_question_if_conclusions_are_equal and df[combo[0]][index] == df[combo[1]][index]:
                same_conclusion_count["--".join(sorted(combo))] = \
                    same_conclusion_count.get("--".join(sorted(combo)), 0) + 1
            else:
                sql_strings.append(
                    "\"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\", \"{}\")".format(
                        df["_topic"][index],
                        clean_premise(df[column_for_premise][index]).replace("\"", "&quot;"),
                        combo[0],
                        df[combo[0]][index].replace("\"", "&quot;"),
                        combo[1],
                        df[combo[1]][index].replace("\"", "&quot;"),
                        df["_specific_frame"][index].replace("\"", "&quot;"),
                        df["_generic_frame"][index].replace("\"", "&quot;"),
                        df["_specific_frame"][index].replace("\"", "&quot;"),
                        df["_generic_frame"][index].replace("\"", "&quot;")
                    )
                )

    sql_string += ", ".join(sql_strings)
    sql_string += ";"

    print(sql_string)

    if skip_question_if_conclusions_are_equal:
        pprint.pprint(object=same_conclusion_count, sort_dicts=True, width=120)