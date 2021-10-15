import math
import pprint
from pathlib import Path
from typing import Optional, List, Tuple, Union

import torch
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoTokenizer, PreTrainedTokenizer, \
    AutoConfig, Trainer, TrainingArguments, default_data_collator

from Frames import FrameSet


class ClassificationDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        return {
            "input_ids": torch.squeeze(self.data[index][0]),
            "labels": self.data[index][1]
        }

    def __len__(self) -> int:
        return len(self.data)

    def __init__(self, data: List[Tuple[Union[str, torch.Tensor], int]],
                 tokenizer: Optional[PreTrainedTokenizer] = None, max_length: Optional[int] = None) -> None:
        if tokenizer is None:
            self.data: List[Tuple[torch.Tensor, int]] = data
        else:
            logger.trace("Tokenizer is given: {}", tokenizer)
            self.data: List[Tuple[torch.Tensor, int]] = [(tokenizer(text, return_tensors="pt", padding="max_length",
                                                                    max_length=max_length)["input_ids"], label)
                                                         if isinstance(text, str) else (text, label)
                                                         for text, label in data]
        logger.success("Initializes a dataset with {} samples", len(self.data))


class FrameClassifier:
    def __init__(self, model: Union[str, Path], frame_set: FrameSet,
                 tokenizer: Optional[Union[str, AutoTokenizer]] = None,
                 train_pairs: Optional[List[Tuple[Union[str, torch.Tensor], int]]] = None, **kwargs) -> None:
        if isinstance(model, Path):
            logger.info("Model is located locally at: {}", model.absolute())
            if model.is_file():
                logger.warning("You have to give the directory, not the specific file. Change model-param to: {}",
                               model.parent)
                model = model.parent

        logger.info("You want to load \"{}\" -- having a fine-tuning in mind: {}", model,
                    "no" if train_pairs is None else "yes")
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model if isinstance(model, str) else str(model.absolute())
        )
        config.num_labels = len(frame_set)
        logger.debug("Load the following config: {}", config)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model if isinstance(model, str) else str(model.absolute()),
            config=config
        )
        logger.success("Successfully loaded a \"{}\" from \"{}\"", type(self.model), model)

        if tokenizer is None:
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model if isinstance(model, str) else str(model.absolute())
            )
            logger.info("Loaded the default tokenizer as well: {}", self.tokenizer)
        else:
            if isinstance(tokenizer, str):
                self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer
                )
                logger.info("Loaded the tokenizer as well: {}", self.tokenizer)
            else:
                logger.debug("Tokenizer is already loaded...")
                self.tokenizer: PreTrainedTokenizer = tokenizer

        self.frame_set: FrameSet = frame_set
        logger.debug("Fetched the frame set: {}", self.frame_set)

        self.max_length = kwargs.get("max_length", 24)

        if train_pairs is not None:
            logger.info("You want to fine-tune \"{}\" with {} samples in total.", type(self.model), len(train_pairs))

            self.out_dir: Path = model if isinstance(model, Path) else Path("frame_classifier",
                                                                            str(self.frame_set), model.replace("/", "_"))

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = logits.argmax(-1)
                return {
                    "acc": accuracy_score(labels, predictions),
                    "balanced_acc": balanced_accuracy_score(labels, predictions)
                }

            self.trainer: Optional[Trainer] = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir=str(self.out_dir.parent.joinpath("train.out", model).absolute()),
                    do_train=True,
                    do_eval=True,
                    do_predict=True,
                    evaluation_strategy="epoch",
                    prediction_loss_only=False,
                    num_train_epochs=7,
                    warmup_steps=100 if len(train_pairs) >= 200 else 1,
                    learning_rate=2e-4 / (1 + math.log10(len(train_pairs))),
                    log_level="info",
                    save_strategy="epoch",
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    label_smoothing_factor=kwargs.get("label_smoothing", .1)
                ),
                compute_metrics=compute_metrics,
                data_collator=default_data_collator,
                train_dataset=ClassificationDataset(
                    data=train_pairs[:int(len(train_pairs) * kwargs.get("train_split", .8))],
                    tokenizer=self.tokenizer if kwargs.get("handle_raw_dataset", False) else None,
                    max_length=self.max_length if kwargs.get("handle_raw_dataset", False) else None
                ),
                eval_dataset=ClassificationDataset(
                    data=train_pairs[int(len(train_pairs) * kwargs.get("train_split", .8)):
                                     int(len(train_pairs) * kwargs.get("val_split", .9))],
                    tokenizer=self.tokenizer if kwargs.get("handle_raw_dataset", False) else None,
                    max_length=self.max_length if kwargs.get("handle_raw_dataset", False) else None
                )
            )

            logger.success("Initialise the trainer: {} ({} samples)", self.trainer, len(self.trainer.train_dataset))
            logger.debug("General args: {}", self.trainer.args)

            self.trainer.train()

            test_dataset = ClassificationDataset(
                data=train_pairs[int(len(train_pairs) * kwargs.get("val_split", .9)):
                                 int(len(train_pairs) * kwargs.get("test_split", 1))],
                tokenizer=self.tokenizer if kwargs.get("handle_raw_dataset", False) else None,
                max_length=self.max_length if kwargs.get("handle_raw_dataset", False) else None
            )
            test_outputs = self.trainer.predict(
                test_dataset=test_dataset
            )

            logger.trace("Received following predictions: {}", test_outputs.predictions)

            test_outputs.metrics["num_samples"] = len(test_dataset)
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                with self.out_dir.joinpath("metrics.txt").open(mode="w", encoding="utf-8") as stream:
                    pprint.pprint(object=test_outputs.metrics,
                                  stream=stream,
                                  indent=4, width=120, depth=3, compact=False)
            except IOError:
                logger.error("Was not able to write a metric-file with the following content from test: {}",
                             pprint.pformat(object=test_outputs.metrics, indent=2, width=100,depth=1, compact=True,
                                            sort_dicts=True))

            self.trainer.save_model(output_dir=str(self.out_dir.absolute()))
            logger.info("Saves the model in \"{}\", so you can use it later with .from_pretrained()", self.out_dir)
        else:
            logger.info("No training samples given -- no additional fine-tuning")
            self.trainer: Optional[Trainer] = None

            self.out_dir: Path = model if isinstance(model, Path) else Path("frame_classifier", str(self.frame_set))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, FrameClassifier) and self.model == o.model

    def __str__(self) -> str:
        return "Classifier for the frame set \"{}\" with the transformer \"{}\"{}".format(
            self.frame_set, type(self.model), "" if self.trainer is None else " (training included)"
        )

    def predict(self, sample: str) -> torch.Tensor:
        logger.debug("You want to predict the probability distribution of the expressed frame(s) for \"{}\"", sample)

        encoding = self.tokenizer(
            text=sample,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=False,
            return_token_type_ids=False,
            return_tensors="pt",
            verbose=False
        )

        logger.trace("Resolved the encoding: {}", encoding)

        if self.model.training:
            logger.warning("The frame-classifier is still in training mode -- let's change this!")
            self.model.eval()

        predictions = self.model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            return_dict=True
        )

        final_predictions = torch.softmax(torch.squeeze(predictions["logits"]), dim=-1)

        logger.debug("The predictions for \"{}\" are: {}", sample, final_predictions)

        return final_predictions


if __name__ == "__main__":
    logger.info("############################# JUST TESTING #############################")
    classifier = FrameClassifier(model="distilbert-base-uncased",
                                 frame_set=FrameSet(frame_set=Path("frame_sets", "neumann_frames.csv"), add_other=False),
                                 train_pairs=[("facts without interpretation", 0),
                                              ("describes effect of an issue to individual or group", 1),
                                              ("the dominance of forces over weak individuals or groups", 2),
                                              ("profit and loss", 3),
                                              ("(most indirectly) morality and social prescriptions", 4),
                                              ("(political) battle!", 5),
                                              ("truths without interpretation", 0),
                                              ("describes effect of an topic to people", 1),
                                              ("outperforming forces over child or gatherings", 2),
                                              ("economic", 3),
                                              ("(most indirectly) morality and social observations", 4),
                                              ("fight - fight! FIGHT!", 5)],
                                 handle_raw_dataset=True)
    logger.success("Your first prediction: {}",
                   classifier.predict(sample="I guess you're right, but should we do this, really? I've my concerns... "
                                             "it's like listening to the devil on my shoulder!"))
