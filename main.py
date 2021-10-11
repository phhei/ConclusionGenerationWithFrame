import math
import pathlib
import sqlite3
import sys
from typing import Tuple, List, Union, Optional

import pandas
import torch
import transformers
from loguru import logger

import Trainer
from Evaluation.Evaluate import score_sql, CherryPicker, CherryPickerSelection
from Frames import FrameSet
from Transformer import FrameBiasedT5ForConditionalGeneration
from const import FRAME_START_TOKEN, FRAME_END_TOKEN, TOPIC_START_TOKEN, TOPIC_END_TOKEN

#INPUT params
datasets: List[pathlib.Path] = [pathlib.Path("Webis-argument-framing.csv")]
frame_datasets: List[Tuple[Optional[pathlib.Path], Tuple[str]]] = \
    [(None, ("frame_index", "input_ids")),
     (None, ("frame_index", "Yinput_ids")),
     (pathlib.Path("frame_sets", "frame_datasets", "Media-frames-immigrationsamesexsmoking.csv"),
      ("frame", "conclusion"))
     ]
limit_samples: int = -1
train_val_test_topic_distinct: bool = limit_samples < 0 or limit_samples >= 200
max_length_premise: int = 128
max_length_conclusion: int = 24
include_frame: bool = True
frame_set: Optional[str] = "media_frames"
include_topic: bool = True

# TRAINING parameters
label_smoothing: Optional[float] = None
tdf_vocab_smoothing_factor: Optional[float] = None
frame_vocab_smoothing_factor: Optional[float] = None
tokenizer: transformers.PreTrainedTokenizer = transformers.T5Tokenizer.from_pretrained("t5-small", extra_ids=128)
if frame_set is None:
    model: transformers.PreTrainedModel = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")
else:
    model: str = "t5-small"
checkpoint: Optional[pathlib.Path] = None
# checkpoint: Optional[pathlib.Path] = pathlib.Path(".out", "pytorch_lightning", "T5ForConditionalGeneration",
#                                                  "128-24", "lightning_logs", "version_3", "checkpoints",
#                                                  "epoch=7-step=2471.ckpt")

# INFERENCE parameters
cherry_picker_selection = CherryPickerSelection()
cherry_pickers: List[str] = [
    "None",
    "BERTPicker",
    "ComprehensivePicker",
    "CheaterPicker"
]
preferred_model_for_frame_identifier = "distilroberta-base"

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


def convert_samples_to_input_str(split: pandas.DataFrame,
                                 frame_start: Optional[str] = FRAME_START_TOKEN,
                                 frame_end: Optional[str] = FRAME_END_TOKEN,
                                 topic_start: Optional[str] = TOPIC_START_TOKEN,
                                 topic_end: Optional[str] = TOPIC_END_TOKEN) -> List[Tuple[str, str, int]]:
    def control_code(row) -> str:
        ret = " "
        if include_frame:
            if cluster_frame is None:
                ret += "{}{}{} ".format("" if frame_start is None else "{} ".format(frame_start),
                                        row["frame"],
                                        "" if frame_end is None else " {}".format(frame_end))
            else:
                ret += "{}{}{} ".format(
                    "" if frame_start is None else "{} ".format(frame_start),
                    cluster_frame.issues_specific_frame_to_generic(
                        issue_specific_frame=row["frame"], topic=row["topic"] if include_topic else None
                    ),
                    "" if frame_end is None else " {}".format(frame_end)
                )
        if include_topic:
            ret += "{}{}{} ".format(
                "" if topic_start is None else "{} ".format(topic_start),
                row["topic"],
                "" if topic_end is None else " {}".format(topic_end)
            )
        return ret.rstrip()
    return [("summarize{}: {}".format(control_code(row), str(row["premise"]).strip(" \"'")), row["conclusion"],
             0 if cluster_frame is None else
             cluster_frame.issues_specific_frame_to_generic(
                 issue_specific_frame=row["frame"], fetch_column=None, topic=row["topic"] if include_topic else None
             ))
            for _, row in split.iterrows()]


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

    train = df[:len_stump_df+int(.8*last_df_size)]
    val = df[int(len_stump_df+.8*last_df_size):int(len_stump_df+.9*last_df_size)]
    if train_val_test_topic_distinct:
        val = make_topic_distinct(train, val)
    test = df[len_stump_df+int(.9*last_df_size):]
    if train_val_test_topic_distinct:
        test = make_topic_distinct(val, test)

    logger.success("Retrieved {} samples: {} train, {} val, {} test", len(train)+len(val)+len(test), len(train),
                   len(val), len(test))

    if frame_set is not None:
        cluster_frame = FrameSet(frame_set=frame_set)

        if "media_frames" not in cluster_frame.name:
            logger.info("You considered the frame set \"{}\" - therefore, let's consider the ecologic frame, too",
                        cluster_frame)
            cluster_frame.add_ecology_frame()
    else:
        cluster_frame = None

    new_special_tokens = {
        "additional_special_tokens": [FRAME_START_TOKEN, FRAME_END_TOKEN, TOPIC_START_TOKEN, TOPIC_END_TOKEN]
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

    val_x, val_y = convert_input_str_to_input_int(convert_samples_to_input_str(val),
                                                  fn_tokenizer=tokenizer,
                                                  max_length=(max_length_premise, max_length_conclusion))
    test_x, test_y = convert_input_str_to_input_int(convert_samples_to_input_str(test),
                                                    fn_tokenizer=tokenizer,
                                                    max_length=(max_length_premise, max_length_conclusion))

    frame_dict = None
    if isinstance(model, str):
        if cluster_frame is None:
            model = transformers.T5ForConditionalGeneration.from_pretrained(model)
            logger.error("You don't define implicitly a transformer model, hence we assume you want to have a frame-"
                         "tailored model. However, you donÄt define a frame set! Fall back to: {}", type(model))
        else:
            logger.warning("We don't have a proper model until yet, only a string \"{}\". "
                           "We assume that the frame-related FrameBiasedT5ForConditionalGeneration is needed.", model)
            frame_dict = cluster_frame.get_frame_count_dict(
                corpora=[[(train_x[fd_column[0]][i].item(),
                            train_y[fd_column[1][1:]][i] if fd_column[1].startswith("Y") else train_x[fd_column[1]][i])
                          for i in range(len(train_x["input_ids"]))] if fd_name is None else
                          [(i[1][fd_column[0]],
                            tokenizer(text=i[1][fd_column[1]], padding=False, truncation=False, return_tensors="pt",
                                      is_split_into_words=False)["input_ids"][0])
                           for i in pandas.read_csv(str(fd_name.absolute()), delimiter="|", verbose=True).iterrows()
                           if "headline" not in i[1][fd_column[0]]]
                         for fd_name, fd_column in frame_datasets],
                vocab_size=len(tokenizer.get_vocab())
            )
            frame_dict_model = \
                {k: torch.log(1+v)/max(torch.max(torch.log(1+v)), math.log(2)) for k, v in frame_dict.items()}
            model = FrameBiasedT5ForConditionalGeneration.from_pretrained(model, frame_dict=frame_dict_model, fast=True,
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
            elif frame_dict is None:
                logger.warning("We didn't compute a frame dictionary which is needed for "
                               "frame_vocab_smoothing_factor = \"{}\"", frame_vocab_smoothing_factor)
            else:
                additional_training_args["frame_words"] = \
                    {f: 1 - frame_vocab_smoothing_factor +
                        (2 * frame_vocab_smoothing_factor * (1 + torch.negative(v/max(1, torch.max(v)))))
                     for f, v in frame_dict.items()}
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
        root_save_path = checkpoint.parent.parent
    trainer.test()

    if len(cherry_pickers) >= 1:
        logger.info("Starting with the inference now. Before we can do this, we must setup the {} cherry-pickers!",
                    len(cherry_pickers))
        chery_tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=preferred_model_for_frame_identifier
        )
        cherry_picker_selection.load_cherrypicker_collection(
            used_frame_set=cluster_frame,
            preferred_model=preferred_model_for_frame_identifier,
            corpus_data=[(chery_tokenizer(text=content["conclusion"], padding="max_length",
                                          max_length=max_length_conclusion, truncation=True, is_split_into_words=False,
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
                                 pathlib.Path("frame_sets", "frame_datasets",
                                              "Media-frames-immigrationsamesexsmoking.csv").absolute()
                             ),
                             delimiter="|",
                             verbose=False,
                             quotechar="\"",
                             doublequote=True
                         ).iterrows()
                         if "headline" not in content["frame"]],
            retrain=False,
            max_length=max_length_conclusion,
            label_smoothing=.1 if label_smoothing is None else label_smoothing,
            handle_raw_dataset=False
        )
        cherry_pickers: List[CherryPicker] = [cherry_picker_selection[s] for s in cherry_pickers]
        logger.debug("You're planning to use: {}", " / ".join(map(lambda cp: str(cp), cherry_pickers)))
        logger.trace("{} cherry-pickers loaded", len(cherry_picker_selection))

    for cherry_picker in cherry_pickers:
        if cherry_picker is not None:
            logger.info("Let's generate data with the the following Cherry-Picker: {}", cherry_picker)
        generated_data = trainer.generate(limit=250, cherry_picker=cherry_picker)
        columns = generated_data.pop("columns")
        df = pandas.DataFrame.from_dict(data=generated_data, orient="index", columns=columns)
        logger.success("Generated {} samples ({})", len(df), "|".join(df.columns))
        if root_save_path is not None:
            root_save_path_cherry_picked = \
                root_save_path.joinpath("beam_search" if cherry_picker is None else cherry_picker.short_str())
            root_save_path_cherry_picked.mkdir(parents=True, exist_ok=True)
            csv_path = root_save_path_cherry_picked.joinpath("predictions.csv")
            sql_path = root_save_path_cherry_picked.joinpath("predictions.sql")
            logger.info("Let's save the generations into {} -> {} / {}", root_save_path_cherry_picked,
                        csv_path.name, sql_path.name)
            df.to_csv(path_or_buf=str(csv_path.absolute()), index_label="Test_ID")
            sql_con = sqlite3.connect(database=str(sql_path.absolute()))
            df.to_sql(name="Predictions", con=sql_con, index_label="Test_ID",
                      if_exists="replace")
            pandas.DataFrame.from_records(
                data=test, index=["test_{}".format(i) for i in range(len(test))]
            ).to_sql(name="Data", con=sql_con, index_label="Test_ID", if_exists="replace")
            sql_con.close()

            if sql_path.exists():
                score_sql(path=sql_path, row_id="Test_ID", col_predict=columns[-1], col_ref=columns[1])
        else:
            logger.warning("Don't save the {} generations because you don't define a saving place", len(generated_data))
