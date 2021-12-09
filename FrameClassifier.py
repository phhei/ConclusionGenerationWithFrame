import math
import pprint
from abc import ABC, abstractmethod
from pathlib import Path
from random import choice, shuffle
from typing import Optional, List, Tuple, Union

import torch
from gensim.models.keyedvectors import KeyedVectors
from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoTokenizer, PreTrainedTokenizer, \
    AutoConfig, Trainer, TrainingArguments, default_data_collator, BatchEncoding

from Frames import FrameSet
from MainUtils import get_glove_w2v_model, stop_words


class ClassificationDataset(Dataset):
    """
    A Dataset for the huggingface-trainer for the generic frame case: a fixed number of classes
    """
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
                                                                    max_length=max_length,
                                                                    truncation=True)["input_ids"], label)
                                                         if isinstance(text, str) else (text, label)
                                                         for text, label in data]
        logger.success("Initializes a dataset with {} samples", len(self.data))


class RegressionDataset(Dataset):
    """
    A Dataset for the huggingface-trainer for the issue-specific frame case: not a fixed number of classes
    """
    def __getitem__(self, index) -> T_co:
        ret = {
            "input_ids": torch.squeeze(self.data[index][0]["input_ids"]),
            "labels": self.data[index][1]
        }
        if "attention_mask" in self.tokenizer.model_input_names:
            ret["attention_mask"] = torch.squeeze(self.data[index][0]["attention_mask"])
        if "token_type_ids" in self.tokenizer.model_input_names:
            ret["token_type_ids"] = torch.squeeze(self.data[index][0]["token_type_ids"])
        else:
            logger.trace("The dataset can't forward the \"token_type_ids\" to the model! "
                         "Maybe not the best model choice for a [SEP]-task? Throw away: {}",
                         torch.squeeze(self.data[index][0]["token_type_ids"]))
        return ret

    def __len__(self) -> int:
        return len(self.data)

    def __init__(self, data: List[Tuple[Tuple[str, str], float]],
                 tokenizer: PreTrainedTokenizer = None, max_length: Optional[int] = None) -> None:
        logger.trace("Tokenizer is given: {}", tokenizer)
        self.tokenizer = tokenizer
        self.data: List[Tuple[BatchEncoding, float]] = \
            [(tokenizer(
                text=text[0], text_pair=text[1], return_tensors="pt", padding="max_length", max_length=max_length,
                return_attention_mask=True, return_token_type_ids=True, truncation="longest_first"
            ), label) for text, label in data]
        logger.success("Initializes a dataset with {} samples", len(self.data))


class FrameClassifier(ABC):
    """
    Base class
    """
    def __init__(self, model: Union[str, Path],
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
            pretrained_model_name_or_path=model if isinstance(model, str) else str(model.absolute()),
            num_labels=kwargs.get("num_labels", 1)
        )
        logger.debug("Load the following config: {}", config)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model if isinstance(model, str) else str(model.absolute()),
            config=config
        )
        logger.success("Successfully loaded a \"{}\" from \"{}\"", self.model.config.model_type, model)

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

        self.max_length = kwargs.get("max_length", 24)

        if train_pairs is None:
            logger.info("No training samples given -- no additional fine-tuning")
            self.trainer: Optional[Trainer] = None

    @abstractmethod
    def train(self, train_pairs: List[Tuple], **kwargs) -> None:
        pass

    def __eq__(self, o: object) -> bool:
        return isinstance(o, FrameClassifier) and self.model == o.model

    def __str__(self) -> str:
        return "General frame classifier with the transformer \"{}\"{}".format(
            self.model.config.model_type, "" if self.trainer is None else " (training included)"
        )

    def predict(self, sample: Union[str, Tuple[str, str]], apply_softmax: bool = True):
        """
        Predicts the score of a specific sample

        :param sample: a sample - mostly a conclusion without any further information - just the plain string
        :param apply_softmax: softmax before return? Makes sense in case of an classification task, but not regression
        :return: a probability-distribution (in case of apply_softmax) or an score value
        """
        if isinstance(sample, Tuple):
            logger.debug("You want to predict how strong \"{}\" is related to the frame \"{}\"", sample[0], sample[1])
        else:
            logger.debug("You want to predict the probability distribution of the expressed frame(s) for \"{}\"",
                         sample)

        encoding = self.tokenizer(
            text=sample if isinstance(sample, str) else sample[0],
            text_pair=None if isinstance(sample, str) else sample[1],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=False,
            return_attention_mask="attention_mask" in self.tokenizer.model_input_names,
            return_token_type_ids=isinstance(sample, Tuple) and "token_type_ids" in self.tokenizer.model_input_names,
            return_length=False,
            return_offsets_mapping=False,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_tensors="pt",
            verbose=False
        )

        logger.trace("Resolved the encoding: {}", encoding)

        if self.model.training:
            logger.warning("The frame-classifier is still in training mode -- let's change this!")
            self.model.eval()

        predictions = self.model(
            **encoding,
            return_dict=True
        )

        final_predictions = torch.squeeze(predictions["logits"])
        if apply_softmax:
            final_predictions = torch.softmax(final_predictions, dim=-1)

        logger.debug("The predictions for \"{}\" are: {}", sample, final_predictions)

        return final_predictions


class GenericFrameClassifier(FrameClassifier):
    def __init__(self, model: Union[str, Path], frame_set: FrameSet,
                 tokenizer: Optional[Union[str, AutoTokenizer]] = None,
                 train_pairs: Optional[List[Tuple[Union[str, torch.Tensor], int]]] = None, **kwargs) -> None:
        self.frame_set: FrameSet = frame_set
        logger.debug("Fetched the frame set: {}", self.frame_set)

        super().__init__(model, tokenizer, train_pairs, num_labels=len(self.frame_set), **kwargs)

        if train_pairs is None:
            self.out_dir: Path = model if isinstance(model, Path) else Path("frame_classifier", str(self.frame_set))
        else:
            self.out_dir: Path = \
                self.model if isinstance(self.model, Path) else Path("frame_classifier",
                                                                     str(self.frame_set),
                                                                     self.model.name_or_path.replace("/", "_"))
            self.train(train_pairs=train_pairs, **kwargs)

    def train(self, train_pairs: List[Tuple[Union[str, torch.Tensor], int]], **kwargs) -> None:
        logger.info("You want to fine-tune \"{}\" ({} classes) with {} samples in total.",
                    self.model.config.model_type, self.model.config.num_labels, len(train_pairs))

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
                output_dir=str(self.out_dir.parent.joinpath("train.out", self.out_dir.name).absolute()),
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
                label_smoothing_factor=kwargs.get("label_smoothing", .1),
                eval_accumulation_steps=100,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32
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
                         pprint.pformat(object=test_outputs.metrics, indent=2, width=100, depth=1, compact=True,
                                        sort_dicts=True))

        self.trainer.save_model(output_dir=str(self.out_dir.absolute()))
        logger.info("Saves the model in \"{}\", so you can use it later with .from_pretrained()", self.out_dir)

    def predict(self, sample: str, apply_softmax: bool = True) -> torch.Tensor:
        if not apply_softmax:
            logger.warning("You have {} classes, you should apply the softmax to get a probability distribution!",
                           len(self.frame_set))
        return super(GenericFrameClassifier, self).predict(sample=sample, apply_softmax=apply_softmax)

    def __str__(self) -> str:
        return "Classifier for the frame set \"{}\" with the transformer \"{}\"{}".format(
            self.frame_set, self.model.config.model_type, "" if self.trainer is None else " (training included)"
        )


class IssueSpecificFrameClassifier(FrameClassifier):
    def __init__(self, model: Union[str, Path], tokenizer: Optional[Union[str, AutoTokenizer]] = None,
                 train_pairs: Optional[List[Tuple[str, str]]] = None, **kwargs) -> None:
        super().__init__(model, tokenizer, train_pairs, num_labels=1, max_length=kwargs.get("max_length", 30),
                         **kwargs)

        self.w2v_model: KeyedVectors = get_glove_w2v_model()

        if train_pairs is None:
            self.out_dir: Path = model if isinstance(model, Path) else Path("frame_classifier", "issue_specific")
        else:
            self.out_dir: Path = \
                self.model if isinstance(self.model, Path) else Path("frame_classifier",
                                                                     "issue_specific",
                                                                     self.model.name_or_path.replace("/", "_"))
            self.train(train_pairs=train_pairs, **kwargs)

    def train(self, train_pairs: List[Tuple[str, str]], **kwargs) -> None:
        logger.info("You want to fine-tune \"{}\" with {} samples in raw.",
                    self.model.config.model_type, len(train_pairs))

        issue_specific_frames = \
            {frame: [t_strip for t in frame.replace("-", " ").replace("/", " ").split(" ")
                     if (t_strip := t.strip(" ,:.?!")) not in stop_words and len(t_strip) > 1]
             for _, frame in train_pairs}
        logger.info("First of all, let's compute a wmd-issue-specific-frame-matrix ({} unique issue-specific frames)",
                    len(issue_specific_frames))
        distance_matrix = \
            {
                frame: {
                    comp_frame: self.w2v_model.wmdistance(document1=frame_split, document2=comp_frame_split)
                    for comp_frame, comp_frame_split in issue_specific_frames.items() if comp_frame != frame
                }
                for frame, frame_split in issue_specific_frames.items()
            }
        logger.trace("Finishing comping the distance-matrix with {} items",
                     sum(map(lambda r: len(r), distance_matrix.values())))

        logger.debug("Enriching the {} train pairs by two counter-samples each", len(train_pairs))

        enriched_train_pairs: List[Tuple[Tuple[str, str], float]] = []
        for conclusion, frame in train_pairs:
            distances = [(comp_frame, distance) for comp_frame, distance in distance_matrix[frame].items()
                         if distance != float("inf")]
            # for i in range(len(distances)):
            #     if distances[i][1] == float("inf"):
            #         comp_frame, _ = distances.pop(i)
            #         logger.warning("There is no WMDistance between {} {} and {} {} -- ignore",
            #                        frame, issue_specific_frames[frame], comp_frame, issue_specific_frames[comp_frame])
            distances.sort(key=lambda d: d[1])
            logger.trace("For \"{}\", the closest issue-specific-frames are {}, the most opposite are {}",
                         distances[:5], distances[-5:])
            enriched_train_pairs.append(((conclusion, frame), 1.))
            try:
                max_distance_frame, max_distance_value = distances.pop()
                logger.debug("\"{}\" <-{}-> \"{}\"", frame, round(max_distance_value, 2), max_distance_frame)
                enriched_train_pairs.append(((conclusion, max_distance_frame), 0.))
                random_pull_frame, random_pull_value = choice(distances)
                enriched_train_pairs.append(((conclusion, random_pull_frame),
                                             1. - random_pull_value / max_distance_value))
            except IndexError:
                logger.opt(exception=False).warning("WMD-calculation failed by \"{}\" ({})", frame,
                                                    issue_specific_frames[frame])
                enriched_train_pairs.append(((conclusion, choice(list(distance_matrix[frame].keys()))), 0.))

        logger.success("Successfully enriched the training data: {}->{} samples, for example: {}",
                       len(train_pairs), len(enriched_train_pairs),
                       ", and ".join(
                           map(lambda etp: "({}|{})-{}".format(etp[0][0], etp[0][1], round(etp[1], 2)),
                               enriched_train_pairs[:min(3, len(enriched_train_pairs))])
                       ))

        train_data = enriched_train_pairs[:int(len(enriched_train_pairs) * kwargs.get("train_split", .8))]
        shuffle(train_data)
        dev_data = enriched_train_pairs[int(len(enriched_train_pairs) * kwargs.get("train_split", .8)):
                                        int(len(enriched_train_pairs) * kwargs.get("val_split", .9))]
        shuffle(dev_data)
        test_data = enriched_train_pairs[int(len(enriched_train_pairs) * kwargs.get("val_split", .9)):
                                         int(len(enriched_train_pairs) * kwargs.get("test_split", 1))]
        shuffle(test_data)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.squeeze(-1)
            three_class_labels = labels * 2
            three_class_predictions = predictions * 2
            return {
                "acc": accuracy_score(three_class_labels.round(), three_class_predictions.round()),
                "error": mean_absolute_error(labels, predictions)
            }

        self.trainer: Optional[Trainer] = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=str(self.out_dir.parent.joinpath("train.out", self.out_dir.name).absolute()),
                do_train=True,
                do_eval=True,
                do_predict=True,
                evaluation_strategy="epoch",
                prediction_loss_only=False,
                num_train_epochs=4,
                warmup_steps=100 if len(enriched_train_pairs) >= 200 else 1,
                learning_rate=3e-4 / (1 + math.log10(len(enriched_train_pairs))),
                log_level="info",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_error",
                greater_is_better=False,
                eval_accumulation_steps=100,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32
            ),
            compute_metrics=compute_metrics,
            data_collator=default_data_collator,
            train_dataset=RegressionDataset(
                data=train_data,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            ),
            eval_dataset=RegressionDataset(
                data=dev_data,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        )

        logger.success("Initialise the trainer: {} ({} samples)", self.trainer, len(self.trainer.train_dataset))
        logger.debug("General args: {}", self.trainer.args)

        self.trainer.train()

        test_dataset = RegressionDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length
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
                         pprint.pformat(object=test_outputs.metrics, indent=2, width=100, depth=1, compact=True,
                                        sort_dicts=True))

        self.trainer.save_model(output_dir=str(self.out_dir.absolute()))
        logger.info("Saves the model in \"{}\", so you can use it later with .from_pretrained()", self.out_dir)

    def __str__(self) -> str:
        return super().__str__()

    def predict(self, sample: Tuple[str, str], apply_softmax: bool = False):
        if apply_softmax:
            logger.warning("It's not sensible to apply softmax on a single value!")
        return super().predict(sample, apply_softmax)


if __name__ == "__main__":
    logger.info("############################# JUST TESTING #############################")
    classifier = GenericFrameClassifier(
        model="distilbert-base-uncased",
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
    logger.info("############################# OK, now issue-specific #############################")
    classifier = IssueSpecificFrameClassifier(
        model="distilbert-base-uncased",
        train_pairs=[("Bert Model transformer with a sequence classification/regression head on top "
                      "(a linear layer on top of the pooled output) e.g. for GLUE tasks.", "inform"),
                     ("This model inherits from PreTrainedModel.", "info"),
                     ("Check the superclass documentation for the generic methods the library implements for all its "
                      "model", "warning"),
                     ("(such as downloading or saving, resizing the input embeddings, pruning heads etc.)", "brackets"),
                     ("This model is also a PyTorch torch.nn.Module subclass.", "programming details"),
                     ("Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter "
                      "related to general usage and behavior.", "usage"),
                     ("Parameters", "code documentation"),
                     ("Indices of input sequence tokens in the vocabulary.", "code"),
                     ("Indices can be obtained using BertTokenizer.", "explanation"),
                     ("What are input IDs?", "question"),
                     ("Mask to avoid performing attention on padding token indices", "code"),
                     ("What are attention masks?", "big question")])
    logger.success("Your first prediction: {}",
                   classifier.predict(sample=("Is there a god?", "strong question")))
