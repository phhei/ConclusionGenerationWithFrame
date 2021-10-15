import math
import pprint
import random
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, \
    BatchEncoding, Trainer, TrainingArguments, default_data_collator

import pandas
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

################################################################################
# ============================== HYPERPARAMETERS ===============================
################################################################################
from typing import Optional

dataset: Path = Path("Webis-argument-framing.csv")
include_topic: bool = True
max_length: int = 128+24
used_portion: Optional[float] = .8
train_part: float = .8
dev_part: float = .1
test_part: float = .1

# ground_model: str = "microsoft/deberta-base-mnli"
ground_model: str = "microsoft/deberta-v2-xlarge-mnli"  #"roberta-large-mnli"
label_smoothing: Optional[None] = .1
max_epochs: int = 5
class_dict: dict = {
    "CONTRADICTION": 0,
    "NEUTRAL": 1,
    "ENTAILMENT": 2
}

#################################################################################
# =================================== PROGRAM ===================================
#################################################################################


class ClassificationDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        ret = {
            "labels": self.y[index]
        }
        ret.update({k: v[index] for k, v in self.x.items()})

        return ret

    def __len__(self) -> int:
        return len(self.y)

    def __init__(self, x: BatchEncoding, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

        assert len(self.x["input_ids"]) == len(self.y)

        logger.success("Initializes a dataset with {} samples", len(self.y))


if __name__ == "__main__":
    df = pandas.read_csv(filepath_or_buffer=str(dataset.absolute()), index_col="argument_id")
    if used_portion is not None:
        df = df[:int(used_portion*len(df))]
    logger.info("Loaded \"{}\": {} samples with following columns: {}", dataset, len(df), df.columns)

    final_dataset = []
    for sid, sample in df.iterrows():
        logger.trace("Fetch following row ({}): {}", sid, sample)
        final_dataset.append(
            ("{}: {}".format(sample["topic"], sample["premise"]) if include_topic
             else sample["premise"],
             sample["conclusion"],
             class_dict["ENTAILMENT"])
        )

        logger.trace("Let's generate a neutral sample (complete random conclusion) for the sample {}", sid)
        neutral_sample = df[df.topic_id != sample["topic_id"]].sample(n=1, replace=False)
        logger.debug("Found following neutral sample for {} (\"{}\"): \"{}\"", sid, sample["premise"],
                     neutral_sample["conclusion"].item())
        final_dataset.append(
            ("{}: {}".format(sample["topic"], sample["premise"]) if include_topic
             else sample["premise"],
             neutral_sample["conclusion"].item(),
             class_dict["NEUTRAL"])
        )

        logger.trace("Let's generate a contradicting sample (complete random conclusion) for the sample {}", sid)
        try:
            contrast_sample = df.query(
                "topic_id == {} and frame_id == {} and stance != '{}'".format(sample["topic_id"], sample["frame_id"],
                                                                              sample["stance"])
            ).sample(n=1, replace=False)
        except ValueError:
            logger.opt(exception=False).warning("Was not able to retrieve a negative sample in the topic \"{}\" "
                                                "with the frame \"{}\"", sample["topic"], sample["frame"])
            try:
                contrast_sample = df.query(
                    "topic_id == {} and stance != '{}'".format(sample["topic_id"], sample["stance"])
                ).sample(n=1, replace=False)
            except ValueError:
                logger.opt(exception=True).error("There is no negative sample at all in thw topic \"{}\" ({}) - "
                                                 "give up...", sample["topic"], sid)
                contrast_sample = None
        if contrast_sample is not None:
            logger.debug("Found following contrastive sample for {} (\"{}\"): \"{}\"", sid, sample["premise"],
                         contrast_sample["conclusion"].item())
            final_dataset.append(
                ("{}: {}".format(sample["topic"], sample["premise"]) if include_topic
                 else sample["premise"],
                 contrast_sample["conclusion"].item(),
                 class_dict["CONTRADICTION"])
            )

    logger.success("Successfully crawled {} samples ({} entail, {} neutral, {} contradiction)",
                   len(final_dataset),
                   len(list(filter(lambda f: f[-1] == class_dict["ENTAILMENT"], final_dataset))),
                   len(list(filter(lambda f: f[-1] == class_dict["NEUTRAL"], final_dataset))),
                   len(list(filter(lambda f: f[-1] == class_dict["CONTRADICTION"], final_dataset))))

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(ground_model)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(ground_model, return_dict=True)

    logger.success("Successfully loaded the model ({}) and tokenizer ({})", model.config.architectures, tokenizer)

    logger.trace("Now let's split the datasets into {}%/{}%/{}%", round(train_part*100), round(dev_part*100),
                 round(test_part*100))

    train = final_dataset[:int(train_part*len(final_dataset))]
    random.shuffle(train)
    dev = final_dataset[int(train_part*len(final_dataset)):int((train_part+dev_part)*len(final_dataset))]
    test = final_dataset[-int(test_part*len(final_dataset)):]

    out_dir: Path = Path("stance_classifier", ground_model, "with topic" if include_topic else "without topic",
                         str(max_length))

    train_x = tokenizer(
        text=list(map(lambda f: f[0], train)),
        text_pair=list(map(lambda f: f[1], train)),
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        add_special_tokens=True,
        is_split_into_words=False,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_offsets_mapping=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_length=False
    )
    dev_x = tokenizer(
        text=list(map(lambda f: f[0], dev)),
        text_pair=list(map(lambda f: f[1], dev)),
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        add_special_tokens=True,
        is_split_into_words=False,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_offsets_mapping=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_length=False
    )
    test_x = tokenizer(
        text=list(map(lambda f: f[0], test)),
        text_pair=list(map(lambda f: f[1], test)),
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        add_special_tokens=True,
        is_split_into_words=False,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_offsets_mapping=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_length=False
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return {
            "acc": accuracy_score(labels, predictions),
            "balanced_acc": balanced_accuracy_score(labels, predictions)
        }

    trainer: Trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir.parent.joinpath("train.out", "verbose").absolute()),
            do_train=True,
            do_eval=True,
            do_predict=True,
            evaluation_strategy="epoch",
            prediction_loss_only=False,
            num_train_epochs=max_epochs,
            warmup_steps=100 if len(train) >= 200 else 1,
            learning_rate=2e-4 / (1 + math.log10(len(train))),
            log_level="info",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            label_smoothing_factor=label_smoothing
        ),
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        train_dataset=ClassificationDataset(
            x=train_x,
            y=torch.LongTensor(list(map(lambda f: f[-1], train)))
        ),
        eval_dataset=ClassificationDataset(
            x=dev_x,
            y=torch.LongTensor(list(map(lambda f: f[-1], dev)))
        )
    )

    logger.success("Initialise the trainer: {} ({} samples)", trainer, len(trainer.train_dataset))
    logger.debug("General args: {}", trainer.args)

    trainer.train()

    test_dataset = ClassificationDataset(
            x=test_x,
            y=torch.LongTensor(list(map(lambda f: f[-1], test)))
        )
    test_outputs = trainer.predict(
        test_dataset=test_dataset
    )

    logger.trace("Received following predictions: {}", test_outputs.predictions)

    test_outputs.metrics["num_samples"] = len(test_dataset)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with out_dir.joinpath("metrics.txt").open(mode="w", encoding="utf-8") as stream:
            pprint.pprint(object=test_outputs.metrics,
                          stream=stream,
                          indent=4, width=120, depth=3, compact=False)
    except IOError:
        logger.error("Was not able to write a metric-file with the following content from test: {}",
                     pprint.pformat(object=test_outputs.metrics, indent=2, width=100, depth=1, compact=True,
                                    sort_dicts=True))

    trainer.save_model(output_dir=str(out_dir.absolute()))
    logger.info("Saves the model in \"{}\", so you can use it later with .from_pretrained()", out_dir)