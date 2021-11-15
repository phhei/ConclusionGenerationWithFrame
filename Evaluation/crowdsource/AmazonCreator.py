import itertools
import pprint
import random
import os

from pathlib import Path

from Evaluation.crowdsource.Utils import combine_generations, translate_generic_frame_readable
from FrameClassifier import FrameSet
from const import TOPIC_END_TOKEN

os.chdir("../../")

csv = [
    ("no_frame_0104", ".out/pytorch_lightning/T5ForConditionalGeneration/128-24/smoothing0.1/idf0.4/t5-large-without-frame/linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv"),
    ("specific_0104", ".out/pytorch_lightning/T5ForConditionalGeneration/128-24/smoothing0.1/idf0.4/t5-large-res/linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv"),
    ("generic_010405", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/128-24/smoothing0.1/tdf0.4/media-frames0.5/t5-large-media-frames/linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv"),
    ("generic_010405_framescore", ".out/pytorch_lightning/FrameBiasedT5ForConditionalGeneration/128-24/smoothing0.1/tdf0.4/media-frames0.5/t5-large-media-frames/linear_regression_cherry_picker/rouge1-rougeL-bertscore_f1-framescore_score/cherry_picked_without-rougebertscore_precisionbertscore_recallbertscore_f1.csv")
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
total_samples = 80
step_size = 3

frame_set = FrameSet(frame_set="media_frames", add_other=True)
#frame_set.add_ecology_frame()


def optimize_input(f_input: str) -> str:
    if TOPIC_END_TOKEN in f_input:
        return f_input[f_input.index(TOPIC_END_TOKEN)+len(TOPIC_END_TOKEN):].lstrip(" :").rstrip()

    if f_input.startswith("summarize:"):
        return f_input[len("summarize:"):].strip()

    return f_input.strip()


if __name__ == "__main__":
    print(str(Path().absolute()))

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

    scalar = int(len(df) / total_samples)
    for survey_id in range(0, total_samples, step_size):
        content = Path("Evaluation", "crowdsource", "AmazonTurkHTML", "header.html").read_text(encoding="utf-8",
                                                                                               errors="ignore")
        for sample_i in range(step_size):
            index = map_increment_to_index[(survey_id+sample_i)*scalar]
            combos = [random.choice([(left, right), (right, left)])
                      for left, right in itertools.combinations(columns_for_comparison, 2)]
            random.shuffle(combos)
            for i, combo in enumerate(combos):
                same_conclusion_count["total"] = same_conclusion_count.get("total", 0) + 1
                if skip_question_if_conclusions_are_equal and df[combo[0]][index] == df[combo[1]][index]:
                    same_conclusion_count["--".join(sorted(combo))] = \
                        same_conclusion_count.get("--".join(sorted(combo)), 0) + 1
                else:
                    content += """
                    <h1>Argument #{0}_{4}: {6}</h1>
                      <div>
                        <crowd-card heading="Premise" style="min-width: 50%; max-width: 85.5%;">
                          <p style="color:blue; padding-left: 8px; padding-right: 5px;">
                            {1}
                          </p>
                        </crowd-card>
                        <br>
                      </div>
                      <div class="side"> 
                        <crowd-card heading="Conclusion 1" id="left{0}_{4}">
                          <p style="color:purple; padding-left: 8px; padding-right: 5px;">
                            {2}
                          </p>
                        </crowd-card>
                      </div>
                      <div class = "middle">
                          <fieldset id="{0}_Validity_{5}">
                            <input type="radio" value="conclusion1" name="{0}_Validity_{5}" required onclick="Validity(true, '{0}_{4}');">
                            <label for="{0}_Validity_{5}"><span style="text-decoration-style: dotted;" title="What makes more sense? &lt;-- Concl. 1 OR Concl. 2 --&gt;">Validity</span></label>
                            <input type="radio" value="conclusion2" name="{0}_Validity_{5}" onclick="Validity(false, '{0}_{4}');">
                          </fieldset>
                          <fieldset id="{0}_Novelty_{5}">
                            <input type="radio" value="conclusion1" name="{0}_Novelty_{5}" required onclick="Novelty(true, '{0}_{4}');">
                            <label for="{0}_Wording_{5}"><span style="text-decoration-style: dotted;" title="More novel stuff &lt;-- Concl. 1 OR Concl. 2 --&gt;">Novelty</span></label>
                            <input type="radio" value="conclusion2" name="{0}_Novelty_{5}" onclick="Novelty(false, '{0}_{4}');">
                          </fieldset>
                          <fieldset id="{0}_FrameSpec_{5}">
                            <input type="radio" value="conclusion1" name="{0}_FrameSpec_{5}" required onclick="FrameSpec(true, '{0}_{4}');">
                            <label for="{0}_Informativeness_{5}"><span style="text-decoration-style: dotted;" title="More {7}? &lt;-- Concl. 1 OR Concl. 2 --&gt;">Perspective &raquo;{7}&laquo;</span></label>
                            <input type="radio" value="conclusion2" name="{0}_FrameSpec_{5}" onclick="FrameSpec(false, '{0}_{4}');">
                          </fieldset>
                    """.format(index, optimize_input(df[column_for_premise][index]), df[combo[0]][index],
                               df[combo[1]][index], i, "__".join(combo),
                               df["_topic"][index], df["_specific_frame"][index], df["_generic_frame"][index])

                    if (df["_specific_frame"][index] != df["_generic_frame"][index]) and \
                            (not skip_generic_frame_question_when_other or df["_generic_frame"][index] != "other"):
                        content += """
                              <fieldset id="{0}_FrmaeGen_{5}">
                                <input type="radio" value="conclusion1" name="{0}_Specificity_{5}" required onclick="FrameGen(true, '{0}_{4}');">
                                <label for="{0}_Specificity_{5}"><span style="text-decoration-style: dotted;" title="More {8}? &lt;-- Concl. 1 OR Concl. 2 --&gt;">Perspective &raquo;{8}&laquo;</span></label>
                                <input type="radio" value="conclusion2" name="{0}_Specificity_{5}" onclick="FrameGen(false, '{0}_{4}');">
                              </fieldset>
                        """.format(index, optimize_input(df[column_for_premise][index]), df[combo[0]][index],
                               df[combo[1]][index], i, "__".join(combo),
                               df["_topic"][index], df["_specific_frame"][index], df["_generic_frame"][index])

                    content += """
                    </div>
                    <div class="side">
                        <crowd-card heading="Conclusion 2" id="right{0}_{4}">
                          <p style="color:purple; padding-left: 8px; padding-right: 5px;">
                            {3}
                          </p>
                        </crowd-card>
                      </div>
                    """.format(index, optimize_input(df[column_for_premise][index]), df[combo[0]][index],
                               df[combo[1]][index], i, "__".join(combo),
                               df["_topic"][index], df["_specific_frame"][index], df["_generic_frame"][index])

        content += Path("Evaluation", "crowdsource", "AmazonTurkHTML", "footer.html").read_text(encoding="utf-8",
                                                                                                errors="ignore")

        write_path = Path(".out", "manual_evaluation", "survey", "AmazonTurk")
        write_path.mkdir(parents=True, exist_ok=True)

        write_path.joinpath("survey_{}.html".format(survey_id)).write_text(data=content, encoding="utf-8")

    if skip_question_if_conclusions_are_equal:
        pprint.pprint(object=same_conclusion_count, sort_dicts=True, width=120)