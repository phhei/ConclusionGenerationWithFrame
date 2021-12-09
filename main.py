import math
from pathlib import Path
import sqlite3
import sys
from typing import Tuple, List, Union, Optional

import pandas
import torch
import transformers
from loguru import logger

import Trainer
import const
from FrameClassifier import GenericFrameClassifier
from Evaluation.Scores.BERTscore import BertScore, BertScorePremConc
from Evaluation.Evaluate import score_matrix
from Evaluation.Scores.GRUENscore import GRUENMetric
from Evaluation.Scores.Rougescore import RougeMetric
from Evaluation.Scores.SurfaceScore import LengthScore, ClaimLikeScore
from Evaluation.Scores.FrameIdentifier import GenericFrameScore, get_generic_frame_classifier, IssueSpecificFrameScore, \
    get_issue_specific_frame_classifier
from Evaluation.Scores.StanceRelationScore import StanceScore
from Frames import FrameSet
from Transformer import FrameBiasedT5ForConditionalGeneration

# INPUT params
datasets: List[Path] = [Path("Webis-argument-framing.csv")]
generic_frame_datasets: List[Tuple[Optional[Path], Tuple[str]]] = \
    [(None, ("frame_index", "input_ids")),
     (None, ("frame_index", "Yinput_ids")),
     (Path("frame_sets", "frame_datasets", "Media-frames-immigrationsamesexsmoking.csv"),
      ("frame", "conclusion"))
     ]
limit_samples: int = int(sys.argv[sys.argv.index("limit_samples") + 1]) if "limit_samples" in sys.argv else -1
train_val_test_topic_distinct: bool = limit_samples < 0 or limit_samples >= 200
max_length_premise: int = 128
max_length_conclusion: int = 24
include_issue_specific_frame: bool = \
    sys.argv[sys.argv.index("include_issue_specific_frame") + 1].upper() == "TRUE" if "include_issue_specific_frame" in sys.argv else True
if include_issue_specific_frame:
    max_length_premise += 4
include_generic_mapped_frame: bool = \
    sys.argv[sys.argv.index("include_generic_mapped_frame") + 1].upper() == "TRUE" if "include_issue_specific_frame" in sys.argv else False
if include_generic_mapped_frame:
    max_length_premise += 4
include_generic_inferred_frame: bool = \
    sys.argv[sys.argv.index("include_generic_inferred_frame") + 1].upper() == "TRUE" if "include_generic_inferred_frame" in sys.argv else False
if include_generic_inferred_frame:
    max_length_premise += 4
frame_set: Optional[str] = sys.argv[sys.argv.index("frame_set")+1] if "frame_set" in sys.argv else None # media_frames
add_ecologic_frame: Optional[bool] = \
    sys.argv[sys.argv.index("add_ecologic_frame") + 1].upper() == "TRUE" if "add_ecologic_frame" in sys.argv else None
include_topic: bool = \
    sys.argv[sys.argv.index("include_topic") + 1].upper() == "TRUE" if "include_topic" in sys.argv else True
if include_topic:
    max_length_premise += 4

# TRAINING parameters
label_smoothing: Optional[float] = \
    float(sys.argv[sys.argv.index("label_smoothing") + 1]) if "label_smoothing" in sys.argv else None
tdf_vocab_smoothing_factor: Optional[float] = \
    float(sys.argv[sys.argv.index("tdf_vocab_smoothing_factor") + 1]) if "tdf_vocab_smoothing_factor" in sys.argv else None
frame_vocab_smoothing_factor: Optional[float] = \
    float(sys.argv[sys.argv.index("frame_vocab_smoothing_factor") + 1]) if "frame_vocab_smoothing_factor" in sys.argv else None
tokenizer: transformers.PreTrainedTokenizer = transformers.T5Tokenizer.from_pretrained("t5-large", extra_ids=128)
model_str: str = sys.argv[sys.argv.index("model")+1] if "model" in sys.argv else "t5-large"
if frame_set is None:
    model: transformers.PreTrainedModel = transformers.T5ForConditionalGeneration.from_pretrained("t5-large")
else:
    model: str = "t5-large"
checkpoint: Optional[Path] = Path(sys.argv[sys.argv.index("checkpoint")+1]) if "checkpoint" in sys.argv else None
# checkpoint: Optional[pathlib.Path] = pathlib.Path(".out", "pytorch_lightning", "T5ForConditionalGeneration",
#                                                  "128-24", "lightning_logs", "version_3", "checkpoints",
#                                                  "epoch=7-step=2471.ckpt")
# checkpoint = pathlib.Path(".out", "pytorch_lightning", "T5ForConditionalGeneration", "128-24", "smoothing0.2", "tdf0.15", "lightning_logs", "version_0", "checkpoints", "epoch=11-step=3707.ckpt")

# INFERENCE parameters
skip_inference: bool = \
    sys.argv[sys.argv.index("skip_inference") + 1].upper() == "TRUE" if "skip_inference" in sys.argv else False
preferred_model_for_frame_identifier: Optional[str] = "distilroberta-base"
preferred_model_for_stance_identifier: Optional[str] = \
    Path("stance_classifier", "microsoft", "deberta-base-mnli", "with topic", "152") \
        if include_topic else \
        Path("stance_classifier", "microsoft", "deberta-base-mnli", "without topic", "152")
preferred_tokenizer_for_stance_identifier: Optional[str] = "microsoft/deberta-base-mnli"
samples_to_be_generate: int = \
    int(sys.argv[sys.argv.index("samples_generate") + 1]) if "samples_generate" in sys.argv else -1
create_for_all_generic_frames: Optional[str] = \
    sys.argv[sys.argv.index("generic_frames_generator") + 1] if "generic_frames_generator" in sys.argv else None

# OTHER PARAMS
log_level_console: str = "INFO"
log_level_file: str = "DEBUG"
log_to_file: bool = False

########################################################################################################################
########################################################################################################################
########################################################################################################################

logger.remove()
logger.add(sink=sys.stdout, level=log_level_console, colorize=True)
if log_to_file:
    logger.add(sink="logs/main_{time}.log", level=log_level_file, colorize=False, rotation="6 MB")


def make_topic_distinct(split_1: pandas.DataFrame, split_2: pandas.DataFrame) -> pandas.DataFrame:
    topics_split1 = set(split_1["topic_id"].astype("str"))
    logger.debug("Found {} distinct topics in split 1: {}", len(topics_split1), topics_split1)
    intersection = split_2.query(expr="topic_id in [{}]".format(",".join(topics_split1)), inplace=False)
    logger.debug("In split 2, {} samples shares those topics - DROP {}", len(intersection), intersection["topic_id"])
    return split_2.drop(index=intersection.index, inplace=False, errors="ignore")


def convert_samples_to_input_str(split: pandas.DataFrame) -> List[Tuple[str, str, int]]:
    def control_code(row) -> Tuple[str, int]:
        ret_control_code = " "
        ret_generic_frame_id = 0
        if include_topic:
            ret_control_code += "{} {} {} ".format(
                const.TOPIC_START_TOKEN,
                row["topic"],
                const.TOPIC_END_TOKEN
            )
        if include_issue_specific_frame:
            ret_control_code += "{} {} {} ".format(
                const.ISSUE_SPECIFIC_FRAME_START_TOKEN,
                row["issue_specific_frame"]
                if "issue_specific_frame" in split.columns and pandas.notna(row["issue_specific_frame"])
                else row["frame"],
                const.ISSUE_SPECIFIC_FRAME_END_TOKEN
            )
        if include_generic_inferred_frame:
            if cluster_frame is None:
                logger.warning("You want to include the inferred generic frame, but you didn't define a frame set! "
                               "Hint: define \"frame_set\"")
            elif generic_frame_classifier_model is None:
                logger.warning("You want to include the inferred generic frame, but you didn't define a frame set! "
                               "Hint: define \"preferred_model_for_frame_identifier\"")
            else:
                try:
                    ret_generic_frame_id = \
                        torch.argmax(generic_frame_classifier_model.predict(sample=row["conclusion"]), dim=-1).item()
                    ret_control_code += "{} {} {} ".format(
                        const.GENERIC_INFERRED_FRAME_START_TOKEN,
                        cluster_frame.data.iloc[ret_generic_frame_id]["keywords_label"],
                        const.GENERIC_INFERRED_FRAME_END_TOKEN
                    )
                except ValueError:
                    logger.opt(exception=True).warning("Failure by extracting the index of the most probable inferred "
                                                       "frame class for including the inferred generic {} frame {}",
                                                       const.GENERIC_INFERRED_FRAME_START_TOKEN,
                                                       const.GENERIC_INFERRED_FRAME_START_TOKEN)
                except KeyError:
                    logger.opt(exception=True).error("Your frame set us malformed - no \"keywords_label\" ({})",
                                                     cluster_frame)
        if include_generic_mapped_frame:
            if cluster_frame is None:
                logger.warning("You want to include the mapped generic frame, but you didn't define a frame set! "
                               "Hint: define \"frame_set\"")
            else:
                ret_control_code += "{} {} {} ".format(
                    const.GENERIC_MAPPED_FRAME_START_TOKEN,
                    cluster_frame.issues_specific_frame_to_generic(
                        issue_specific_frame=row["frame"], topic=row["topic"] if include_topic else None
                    ),
                    const.GENERIC_MAPPED_FRAME_END_TOKEN
                )
                ret_generic_frame_id = cluster_frame.issues_specific_frame_to_generic(
                    issue_specific_frame=row["frame"], fetch_column=None, topic=row["topic"] if include_topic else None
                )
        return ret_control_code.rstrip(), ret_generic_frame_id
    return [("summarize{}: {}".format(c_out[0], str(row["premise"]).strip(" \"'")), row["conclusion"], c_out[1])
            for _, row in split.iterrows() if (c_out := control_code(row)) is not None]


def convert_input_str_to_input_int(split: List[Tuple[str, str]], fn_tokenizer: transformers.PreTrainedTokenizer,
                                   max_length: Union[int, Tuple[int, int]]) \
        -> Tuple[transformers.BatchEncoding, transformers.BatchEncoding]:
    max_length_x = max_length if isinstance(max_length, int) else max_length[0]
    if max_length_x >= 1:
        logger.info("You want to restrict the token (length) of the input to {}", max_length_x)
    max_length_y = max_length if isinstance(max_length, int) else max_length[1]
    if max_length_y >= 1:
        logger.info("You want to restrict the token (length) of the output to {}", max_length_y)

    x = fn_tokenizer(text=[s[0] for s in split],
                     add_special_tokens=True,
                     truncation="do_not_truncate" if max_length_x <= 0 else "longest_first",
                     padding="longest" if max_length_x <= 0 else "max_length",
                     max_length=None if max_length_x <= 0 else max_length_x,
                     return_tensors="pt",
                     return_token_type_ids=True,
                     return_attention_mask=True,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False,
                     return_offsets_mapping=False,
                     return_length=True,
                     verbose=True)
    x["frame_index"] = torch.IntTensor([s[2] for s in split])
    y = fn_tokenizer(text=[s[1] for s in split],
                     add_special_tokens=True,
                     truncation="do_not_truncate" if max_length_y <= 0 else "longest_first",
                     padding="longest" if max_length_x <= 0 else "max_length",
                     max_length=None if max_length_y <= 0 else max_length_y,
                     return_tensors="pt",
                     return_token_type_ids=True,
                     return_attention_mask=True,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False,
                     return_offsets_mapping=False,
                     return_length=True,
                     verbose=False)
    return x, y


if __name__ == '__main__':
    logger.info("Let's start our argument generation - base datasets: {}", ", ".join(map(lambda ds: ds.name, datasets)))

    df = pandas.DataFrame()
    last_df_size = 0
    for d in datasets:
        last_df = pandas.read_csv(filepath_or_buffer=str(d.absolute()), index_col="argument_id")
        last_df_size = len(last_df)
        logger.debug("Read \"{}\": {} lines", d.name, len(last_df))
        try:
            df = df.append(other=last_df, ignore_index=False, verify_integrity=True, sort=False)
        except ValueError:
            logger.opt(exception=True).error("Failing to concatenate \"{}\": duplicated indices", d.absolute())

    logger.debug("Retrieved {} samples from {} (header: {})", len(df), datasets, " - ".join(df.columns))
    logger.trace(df)

    if limit_samples >= 1:
        logger.info("You want to limit your samples to {}", limit_samples)
        if limit_samples < 10:
            logger.error("You want to limit your samples to only {}. That's too few - please consider >= 10!",
                         limit_samples)
            limit_samples = 10
        if limit_samples < len(df):
            df = df[-limit_samples:]
            last_df_size = min(last_df_size, limit_samples)
        else:
            logger.error("You have fewer samples than your limit_samples = {}", limit_samples)

    len_stump_df = len(df) - last_df_size

    train: pandas.DataFrame = df[:len_stump_df+int(.8*last_df_size)]
    val: pandas.DataFrame = df[int(len_stump_df+.8*last_df_size):int(len_stump_df+.9*last_df_size)]
    if train_val_test_topic_distinct:
        val = make_topic_distinct(train, val)
    test: pandas.DataFrame = df[len_stump_df+int(.9*last_df_size):]
    if train_val_test_topic_distinct:
        test = make_topic_distinct(val, test)

    logger.success("Retrieved {} samples: {} train, {} val, {} test", len(train)+len(val)+len(test), len(train),
                   len(val), len(test))

    if frame_set is not None:
        cluster_frame = FrameSet(frame_set=frame_set)

        if (add_ecologic_frame is None and len(cluster_frame) <= 10) or add_ecologic_frame:
            logger.info("You considered the frame set \"{}\" with {} frames - "
                        "therefore, let's consider the ecologic frame, too",
                        cluster_frame.name, len(cluster_frame))
            cluster_frame.add_ecology_frame()
    else:
        cluster_frame = None

    if cluster_frame is not None and preferred_model_for_frame_identifier is not None:
        frame_tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=preferred_model_for_frame_identifier
        )
        generic_frame_classifier_model: GenericFrameClassifier = get_generic_frame_classifier(
            frame_set=cluster_frame,
            preferred_model=preferred_model_for_frame_identifier,
            corpus_data=[(frame_tokenizer(text=content["conclusion"], padding="max_length",
                                          max_length=max_length_conclusion, truncation=True,
                                          is_split_into_words=False,
                                          return_tensors="pt")["input_ids"][0],
                          cluster_frame.issues_specific_frame_to_generic(
                              issue_specific_frame=content["frame"],
                              topic=str(content["argument_id"])[:str(content["argument_id"]).index("1")],
                              fetch_column=None,
                              semantic_reordering=False,
                              remove_stopwords=True
                          ) if cluster_frame is not None else int(content["frame_id"]))
                         for index, content in
                         pandas.read_csv(
                             filepath_or_buffer=str(
                                 Path("frame_sets", "frame_datasets",
                                      "Media-frames-immigrationsamesexsmoking.csv").absolute()
                             ),
                             delimiter="|",
                             verbose=False,
                             quotechar="\"",
                             doublequote=True
                         ).iterrows()
                         if "headline" not in content["frame"]
                         ],
            retrain=False,
            max_length=max_length_conclusion,
            label_smoothing=.1 if label_smoothing is None else label_smoothing,
            handle_raw_dataset=False
        )
    else:
        generic_frame_classifier_model = None

    new_special_tokens = {
        "additional_special_tokens": [
            const.TOPIC_START_TOKEN,
            const.TOPIC_END_TOKEN,
            const.ISSUE_SPECIFIC_FRAME_START_TOKEN,
            const.ISSUE_SPECIFIC_FRAME_END_TOKEN,
            const.GENERIC_MAPPED_FRAME_START_TOKEN,
            const.GENERIC_MAPPED_FRAME_END_TOKEN,
            const.GENERIC_INFERRED_FRAME_START_TOKEN,
            const.GENERIC_INFERRED_FRAME_END_TOKEN
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens_dict=new_special_tokens)
    logger.info("Added the following {} special tokens: {} (vocab size is {} now)",
                num_added,
                " :: ".join(tokenizer.get_added_vocab().keys()),
                len(tokenizer.get_vocab()))

    train_x, train_y = convert_input_str_to_input_int(split=convert_samples_to_input_str(train),
                                                      fn_tokenizer=tokenizer,
                                                      max_length=(max_length_premise, max_length_conclusion))
    length_x = train_x.pop("length")
    length_y = train_y.pop("length")
    max_length_premise = torch.max(length_x).item()
    logger.info("Your premises have a length of {} on average ({}-{})",
                round(torch.sum(length_x, dtype=torch.float).item()/len(length_x), 2), torch.min(length_x).item(),
                max_length_premise)
    max_length_conclusion = torch.max(length_y).item()
    logger.info("Your conclusions have a length of {} on average ({}-{})",
                round(torch.sum(length_y, dtype=torch.float).item() / len(length_y), 1), torch.min(length_y).item(),
                max_length_conclusion)
    if cluster_frame is not None and "frame_index" in train_x:
        logger.info("And you have the following frame-distribution (in the training data, clustered with {}): {}",
                    cluster_frame,
                    ", ".join(["{}: {}".format(c[1], (train_x["frame_index"] == c[0]).count_nonzero().item())
                               for c in cluster_frame.data.itertuples(index=True, name=None)]))

    val_x, val_y = convert_input_str_to_input_int(split=convert_samples_to_input_str(val),
                                                  fn_tokenizer=tokenizer,
                                                  max_length=(max_length_premise, max_length_conclusion))
    alternating_frame_set: Optional[FrameSet] = None
    if create_for_all_generic_frames is not None:
        logger.info("You want to create a frame-generation matrix on the base of \"{}\" - fine, let's load it!",
                    create_for_all_generic_frames)
        alternating_frame_set = FrameSet(frame_set=create_for_all_generic_frames, add_other=False)
        logger.warning("Using \"{}\" as base will increase your test size (and hence, the generation run-time) "
                       "by {} times!", alternating_frame_set, len(alternating_frame_set))
        test_extend = pandas.DataFrame(columns=test.columns)
        for row_id, data in test.iterrows():
            logger.trace("Duplicating row \"{}\" now...", row_id)
            try:
                def apply_frame(series: pandas.Series, frame_id, frame) -> pandas.Series:
                    s = series.copy(deep=True)
                    s["issue_specific_frame_id"] = s["frame_id"]
                    s["issue_specific_frame"] = s["frame"]
                    s["frame_id"] = frame_id
                    s["frame"] = frame
                    return s

                test_extend = test_extend.append(
                    pandas.DataFrame(data=[data], columns=test.columns, index=[row_id]),
                    ignore_index=False, verify_integrity=True
                ).append(
                    pandas.DataFrame(
                        data=[apply_frame(series=data, frame_id=frame_id, frame=frame["label"])
                              for frame_id, frame in alternating_frame_set.data.iterrows()],
                        index=["{}_{}".format(row_id, i) for i in range(len(alternating_frame_set))],
                        columns=list(test.columns) + ["issue_specific_frame_id", "issue_specific_frame"]),
                    ignore_index=False, verify_integrity=True
                )
            except ValueError:
                logger.opt(exception=True).error("Something bad happened - ignore row \"{}\"", row_id)
        logger.success("Successfully enriched the test set: {} --> {} rows", len(test), len(test_extend))
        test = test_extend
    test_x, test_y = convert_input_str_to_input_int(split=convert_samples_to_input_str(test),
                                                    fn_tokenizer=tokenizer,
                                                    max_length=(max_length_premise, max_length_conclusion))

    generic_frame_dict = None
    if isinstance(model, str):
        if cluster_frame is None:
            model = transformers.T5ForConditionalGeneration.from_pretrained(model)
            logger.error("You don't define implicitly a transformer model, hence we assume you want to have a frame-"
                         "tailored model. However, you don't define a frame set! Fall back to: {}", type(model))
        else:
            logger.warning("We don't have a proper model until yet, only a string \"{}\". "
                           "We assume that the frame-related FrameBiasedT5ForConditionalGeneration is needed.", model)
            generic_frame_dict = cluster_frame.get_frame_count_dict(
                corpora=[[(train_x[fd_column[0]][i].item(),
                           train_y[fd_column[1][1:]][i] if fd_column[1].startswith("Y") else train_x[fd_column[1]][i])
                          for i in range(len(train_x["input_ids"]))] if fd_name is None else
                          [(i[1][fd_column[0]],
                            tokenizer(text=i[1][fd_column[1]], padding=False, truncation=False, return_tensors="pt",
                                      is_split_into_words=False)["input_ids"][0])
                           for i in pandas.read_csv(str(fd_name.absolute()), delimiter="|", verbose=True).iterrows()
                           if "headline" not in i[1][fd_column[0]]]
                         for fd_name, fd_column in generic_frame_datasets],
                vocab_size=len(tokenizer.get_vocab())
            )
            frame_dict_model = \
                {k: torch.log(1+v)/max(torch.max(torch.log(1+v)), math.log(2)) for k, v in generic_frame_dict.items()}
            model = FrameBiasedT5ForConditionalGeneration.from_pretrained(model,
                                                                          frame_dict=frame_dict_model, fast=True,
                                                                          sequence_length=max_length_conclusion)

    model.resize_token_embeddings(new_num_tokens=len(tokenizer.get_vocab()))

    if checkpoint is None:
        additional_training_args = None
        if label_smoothing is not None:
            additional_training_args = {"label_smoothing": label_smoothing}
        if tdf_vocab_smoothing_factor is not None:
            if additional_training_args is None:
                logger.error("You can't have a proper tdf-smoothing (tdf_vocab_smoothing_factor: {}) when you ignore "
                             "the label_smoothing in general! Please set a value for label_smoothing",
                             tdf_vocab_smoothing_factor)
            else:
                tdf = torch.stack([(train_x["input_ids"] == i).count_nonzero() for i in range(len(tokenizer.get_vocab()))])
                logger.debug("Calculated the term-frequency: {} (most used token is \"{}\", was used {}x)", tdf,
                             tokenizer.convert_ids_to_tokens(ids=torch.argmax(tdf).item(), skip_special_tokens=False),
                             torch.max(tdf))
                tdf_log = torch.log10(tdf+1)
                additional_training_args["tdf"] = 1 - tdf_vocab_smoothing_factor + \
                                                  (2 * tdf_vocab_smoothing_factor * (tdf_log/max(1, torch.max(tdf_log))))
                logger.trace("Computed the additional_training_args out of {}: {}", train_x["input_ids"].shape,
                             additional_training_args["tdf"])
        if frame_vocab_smoothing_factor is not None and additional_training_args is not None:
            if additional_training_args is None:
                logger.error("You can't have a proper frame-smoothing (frame_vocab_smoothing_factor: {}) "
                             "when you ignore the label_smoothing in general! Please set a value for label_smoothing",
                             frame_vocab_smoothing_factor)
            elif generic_frame_dict is None:
                logger.warning("We didn't compute a frame dictionary which is needed for "
                               "frame_vocab_smoothing_factor = \"{}\"", frame_vocab_smoothing_factor)
            else:
                additional_training_args["frame_words"] = \
                    {f: 1 - frame_vocab_smoothing_factor +
                        (2 * frame_vocab_smoothing_factor * (1 + torch.negative(v/max(1, torch.max(v)))))
                     for f, v in generic_frame_dict.items()}
                logger.debug("The frame_vocab_smoothing_dict was calculated. Contains {} entries.",
                             len(additional_training_args["frame_words"]))
                if -1 not in additional_training_args["frame_words"].keys():
                    logger.warning("frame_vocab_smoothing_dict misses the default entry -1. Contains only: {}",
                                   ", ".join(map(lambda e: str(e),  additional_training_args["frame_words"].keys())))
                    additional_training_args["frame_words"][-1] = \
                        torch.ones((len(tokenizer.get_vocab()),), dtype=torch.float)
        trainer = Trainer.T5Trainer(tokenizer=tokenizer, model=model, data_x=(train_x, val_x, test_x),
                                    data_y=(train_y, val_y, test_y), additional_training_args=additional_training_args)
    else:
        trainer = Trainer.T5Trainer.from_checkpoint(checkpoint=checkpoint, data_x=(train_x, val_x, test_x),
                                                    data_y=(train_y, val_y, test_y), raw_model=model)

    if checkpoint is None:
        root_save_path = trainer.train()
    else:
        root_save_path = checkpoint.parent.parent.parent.parent.joinpath(checkpoint.parent.parent.name)
    trainer.test()

    if skip_inference:
        logger.warning("You want to skip the inference, hence: no generations! Quiet the app here. If you want to "
                       "continue, please call this app again with a \"checkpoint\"-param pointing to \"{}\"",
                       trainer.trainer_module.checkpoint.best_model_path
                       if trainer.trainer_module is not None else "NOT SAVED")
        exit(0)

    logger.trace("###################################################################################################")
    logger.trace("####################################### Initializes Metrics #######################################")
    logger.trace("###################################################################################################")

    metrics_list = [
        BertScore(),
        BertScorePremConc(only_precision=False),
        GRUENMetric(),
        RougeMetric(),
        LengthScore(include_premise=True, filter_stopwords=True),
        ClaimLikeScore()
    ]
    if cluster_frame is not None and generic_frame_classifier_model is not None:
        metrics_list.append(
            GenericFrameScore(
                frame_set=cluster_frame,
                frame_classifier=generic_frame_classifier_model
            )
        )
        logger.success("Appended a FrameScorer: {} (base: {})", metrics_list[-1], preferred_model_for_frame_identifier)
    elif cluster_frame is not None:
        logger.warning("You use a generic frame cluster ({}), but you want not to check the frame evaluation scores!",
                       cluster_frame.name)
    if include_issue_specific_frame and preferred_model_for_frame_identifier is not None:
        try:
            corpus_data = \
                [(row["conclusion"],
                  row["issue_specific_frame"]
                  if "issue_specific_frame" in row and pandas.notna(row["issue_specific_frame"])
                  else row["frame"])
                 for _, row in train.iterrows()]
        except KeyError:
            logger.opt(exception=True).warning("Can't extract training data for the issue-specific-frame-regressor")
            corpus_data = None
        metrics_list.append(IssueSpecificFrameScore(
            frame_classifier=get_issue_specific_frame_classifier(
                preferred_model=preferred_model_for_frame_identifier,
                corpus_data=corpus_data
            )
        ))
    if preferred_model_for_stance_identifier is not None:
        metrics_list.append(
            StanceScore(
                stance_classifier=preferred_model_for_stance_identifier,
                classifier_tokenizer=preferred_tokenizer_for_stance_identifier,
                include_topic=include_topic
            )
        )
        logger.success("Appended a StanceScorer: {} (base: {})", metrics_list[-1],
                       preferred_model_for_stance_identifier)
    else:
        logger.warning("You deny a proper stance scorer!")

    logger.trace("####################################################################################################")
    logger.trace("######################################### Start  inference #########################################")
    logger.trace("####################################################################################################")

    generated_data = trainer.generate(
        limit=samples_to_be_generate, cherry_picker=None, comprehensive_result=True,
        alternating_index=1 if alternating_frame_set is None else 1+len(alternating_frame_set)
    )
    columns = generated_data.pop("columns")
    logger.success("Generated {} samples ({})", len(generated_data), "|".join(columns))

    alternating_frame_collection = [""]
    if alternating_frame_set is not None:
        alternating_frame_collection.extend(
            [name["label"] for _, name in alternating_frame_set.data.iterrows()]
        )
    for alternating_frame_index, alternating_frame_name in enumerate(alternating_frame_collection):
        logger.info("Let's collect and analyse the data for frame {}",
                    "ORIGINAL" if alternating_frame_name == "" else alternating_frame_name)
        current_dict = {k[:-(len(str(alternating_frame_index))+1)]: v
                        for k, v in generated_data.items() if k.endswith("_{}".format(alternating_frame_index))}
        df = pandas.DataFrame.from_dict(
            data=current_dict,
            orient="index", columns=columns)
        file_ending = "{}{}".format(
            "_{}".format(samples_to_be_generate) if samples_to_be_generate >= 1 else "",
            "_{}".format(alternating_frame_name.replace(",", "").replace(" ", "-"))
            if len(alternating_frame_name) >= 1 else ""
        )
        logger.debug("{} -> {} ({})", len(generated_data), len(current_dict), file_ending)

        if root_save_path is not None:
            root_save_path.mkdir(parents=True, exist_ok=True)
            try:
                df.to_csv(
                    path_or_buf=root_save_path.joinpath("predictions{}.csv".format(file_ending)),
                    index_label="test_ID",
                    encoding="utf-8",
                    errors="replace"
                )
                sql_con = sqlite3.connect(database=str(root_save_path.joinpath("predictions.sql").absolute()))
                df.to_sql(name="Predictions{}".format(file_ending), con=sql_con, index_label="Test_ID", if_exists="replace")
                pandas.DataFrame.from_records(
                    data=test, index=["test_{}".format(i) for i in range(len(test))]
                ).to_sql(name="Data", con=sql_con, index_label="Test_ID", if_exists="replace")
                sql_con.close()
            except sqlite3.OperationalError:
                logger.opt(exception=True).warning("Was not able to write the SQL-Database (not up to date)")
            except IOError:
                logger.opt(exception=True).error("Was not able to write the CSV-File for the generated conclusions. "
                                                 "Print them into the log: {}", df.to_string())

            score_matrix(ret_dict=current_dict, evaluation_instances=metrics_list)
            columns_extended = list(current_dict[list(current_dict.keys())[0]].keys())
            df = pandas.DataFrame.from_dict(data=current_dict, orient="index", columns=columns_extended)
            try:
                df.to_csv(path_or_buf=root_save_path.joinpath("predictions_scores{}.csv".format(file_ending)),
                          index_label="test_ID", encoding="utf-8", errors="replace")
                sql_con = sqlite3.connect(database=str(root_save_path.joinpath("predictions_scores.sql").absolute()))
                df.to_sql(name="Predictions{}".format(file_ending), con=sql_con, index_label="Test_ID", if_exists="replace")
                pandas.DataFrame.from_records(
                    data=test, index=["test_{}".format(i) for i in range(len(test))]
                ).to_sql(name="Data", con=sql_con, index_label="Test_ID", if_exists="replace")
                sql_con.close()

                logger.success("Successfully saved the results of {} samples here in this dictionary: {}", len(df),
                               root_save_path.absolute())
            except sqlite3.OperationalError:
                logger.opt(exception=True).info("Was not able to write the SQL-Database (not up to date)")
            except IOError:
                logger.opt(exception=True).warning("Was not able to write the CSV-File for the hot cherries. "
                                                   "Print them into the log: {}", df.to_string())
        else:
            logger.warning("Don't save the {} generations because you don't define a saving place", len(current_dict))

