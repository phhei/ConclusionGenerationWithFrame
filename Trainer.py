import pathlib
from typing import Any, Optional, Tuple, Dict, Union

import loguru
import pytorch_lightning
import torch
import torch.nn.functional
import transformers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from Evaluation.Evaluate import CherryPicker
from Transformer import FrameBiasedT5ForConditionalGeneration

logger = loguru.logger


class T5Trainer:
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model: Union[transformers.PreTrainedModel, pathlib.Path],
                 data_x: Tuple[transformers.BatchEncoding, transformers.BatchEncoding, transformers.BatchEncoding],
                 data_y: Tuple[transformers.BatchEncoding, transformers.BatchEncoding, transformers.BatchEncoding],
                 additional_training_args: Optional[Dict] = None):
        self.tokenizer = tokenizer
        self.model = model
        self.train_x = data_x[0]
        self.train_y = data_y[0]
        self.val_x = data_x[1]
        self.val_y = data_y[1]
        self.test_x = data_x[2]
        self.test_y = data_y[2]

        self.trainer_module = None
        self.teacher = None
        self.additional_training_args = additional_training_args

        logger.success("Initialize the {}-Trainer with {} training data-points",
                       "{}[...]{}".format(str(self.model)[0:30], str(self.model)[-12:]),
                       len(self.train_x["input_ids"]))

    @classmethod
    def from_checkpoint(cls, checkpoint: pathlib.Path,
                        data_x: Tuple[transformers.BatchEncoding,
                                      transformers.BatchEncoding,
                                      transformers.BatchEncoding],
                        data_y: Tuple[transformers.BatchEncoding,
                                      transformers.BatchEncoding,
                                      transformers.BatchEncoding],
                        raw_model: Optional[transformers.PreTrainedTokenizer]):
        logger.debug("Recognize checkpoint \"{}\"...", checkpoint.name)
        trainer_module = cls.Training.load_from_checkpoint(checkpoint_path=checkpoint, raw_model=raw_model)
        logger.success("Loaded training core: {}", trainer_module)

        ret = cls(tokenizer=trainer_module.t_tokenizer, model=trainer_module.model, data_x=data_x, data_y=data_y)

        logger.debug("Teacher is missing")

        root_dir = pathlib.Path(".out", "checkpoint", type(trainer_module.model).__name__, checkpoint.stem)
        ret.teacher = \
            pytorch_lightning.Trainer(check_val_every_n_epoch=1, min_epochs=2, max_epochs=12,
                                      log_every_n_steps=8, flush_logs_every_n_steps=16,
                                      progress_bar_refresh_rate=1,
                                      default_root_dir=str(root_dir.absolute()),
                                      # https://github.com/PyTorchLightning/pytorch-lightning/issues/5604
                                      **(dict()
                                         if isinstance(trainer_module.model, FrameBiasedT5ForConditionalGeneration) else
                                         {"num_processes": 2,
                                          "plugins": DDPPlugin(find_unused_parameters=True)}))

        return ret

    def __str__(self) -> str:
        if self.additional_training_args is not None:
            return "Trainer({},{}, additional_training_args: [{}])".format(
                self.model, self.tokenizer,
                ", ".join(
                    map(
                        lambda x: "{}:{}".format(
                            x[0], x[1] if isinstance(x[1], int) else
                            (round(x[1], 1) if isinstance(x[1], float) else "Object")
                        ), self.additional_training_args.items()))
            )
        else:
            return "Trainer({},{})".format(self.model, self.tokenizer)

    class PermuteCrossEntropy(torch.nn.CrossEntropyLoss):
        def forward(self, ce_input: Tensor, ce_target: Tensor) -> Tensor:
            return super().forward(ce_input.permute(0, 2, 1), ce_target)

    class CustomCrossEntropyLoss(torch.nn.Module):
        def __init__(self, vocabulary_size: int, label_smoothing: float = 0.,
                     tdf_vocab_smoothing_factor: Optional[torch.Tensor] = None,
                     dict_frame_vocab_smoothing_factor: Optional[Dict[int, torch.Tensor]] = None):
            """
            Initializes a more fancy cross entropy loss function.

            :param vocabulary_size: the expected vector length per prediction (token).
            :param label_smoothing: a label smoothing factor. The given float decreases the ONE-hot and distributes it
                                    to the other vocabularies
            :param tdf_vocab_smoothing_factor: a singe vector with an float for each vocab, multiplied with the
                                                plain smoothing vector
                                                1 means: don't touch the smoothing!
                                                <1 means: "harden the curve, be more sure in predicting exact
                                                            this expected vocab token!"
                                                >1 means: "soften the curve, don't be too sure (rewarded),
                                                            please allow alternative tokens than exact this, too!"
            :param dict_frame_vocab_smoothing_factor: a dictionary of the form
                                                        frame index -> tdf_vocab_smoothing_factor.
                                                        the mult-smoothing vector is multiplied by the already
                                                        preprocessed smoothing vector (hence, after regular
                                                        smoothing and tdf-smoothing). Hence, a 1 means no change,
                                                        <1 means "harden the curve, this is frame-critical".
                                                        >1 means "soften the curve, regularize more!"
            """
            super().__init__()
            self.vocabulary_size = vocabulary_size
            self.label_smoothing = label_smoothing
            self.label_smoothing_plus = label_smoothing/vocabulary_size
            self.label_smoothing_minus = label_smoothing
            self.dict_tdf_vocab_smoothing_factor = tdf_vocab_smoothing_factor
            self.dict_frame_vocab_smoothing_factor = dict_frame_vocab_smoothing_factor

        def smooth_target(self, inputs_type, targets: torch.Tensor,
                          frame_id: Optional[torch.Tensor] = None) -> torch.Tensor:
            def scale_op_tensor(op_tensor: torch.Tensor,
                                scale_lookup: Union[torch.Tensor, Dict[int, torch.Tensor]]) -> torch.Tensor:
                try:
                    ret_v = []
                    for s_i, sample_smooth in enumerate(op_tensor):
                        ret_s = []
                        # noinspection PyUnresolvedReferences
                        current_lookup = \
                            scale_lookup.get(0 if frame_id is None else frame_id[s_i].item(), scale_lookup[-1])\
                                if isinstance(scale_lookup, Dict) else scale_lookup
                        for i, sample_smooth_token in enumerate(sample_smooth):
                            ret_s.append(sample_smooth_token * current_lookup[targets[s_i][i]])
                        ret_v.append(torch.stack(tensors=ret_s))
                    return torch.stack(tensors=ret_v)
                except KeyError:
                    logger.opt(exception=False).warning("It's mandatory to provide the key \"-1\" (default) in the "
                                                        "frame-scaling-dict!")
                except Exception:
                    logger.opt(exception=True).error(
                        "Unexpected inputs (op_tensor: {} scale_lookup: {}). Notice: in case of tdf, "
                        "you need a 1-dim tensor. The one dimension provides for each vocabulary index ({} in total) "
                        "a scale factor, in case of frame, each of this scaling vector is a value in a dict, "
                        "distinguishing between the current frames",
                        op_tensor.shape,
                        scale_lookup.shape if isinstance(scale_lookup, torch.Tensor) else type(scale_lookup),
                        self.vocabulary_size
                    )

                return op_tensor

            one_hot_targets = \
                torch.nn.functional.one_hot(targets, num_classes=self.vocabulary_size).type(inputs_type)
            add_tensor = self.label_smoothing_plus \
                if self.dict_tdf_vocab_smoothing_factor is None and self.dict_frame_vocab_smoothing_factor is None else\
                torch.full_like(input=one_hot_targets, fill_value=self.label_smoothing_plus)
            minus_tensor = self.label_smoothing_minus \
                if self.dict_tdf_vocab_smoothing_factor is None and self.dict_frame_vocab_smoothing_factor is None else\
                torch.full_like(input=one_hot_targets, fill_value=self.label_smoothing_minus)

            if self.dict_tdf_vocab_smoothing_factor is not None:
                add_tensor = scale_op_tensor(op_tensor=add_tensor,
                                             scale_lookup=self.dict_tdf_vocab_smoothing_factor)
                minus_tensor = scale_op_tensor(op_tensor=minus_tensor,
                                               scale_lookup=self.dict_tdf_vocab_smoothing_factor)
            if self.dict_frame_vocab_smoothing_factor is not None:
                add_tensor = scale_op_tensor(op_tensor=add_tensor,
                                             scale_lookup=self.dict_frame_vocab_smoothing_factor)
                minus_tensor = scale_op_tensor(op_tensor=minus_tensor,
                                               scale_lookup=self.dict_frame_vocab_smoothing_factor)

            one_hot_targets = torch.clip(input=one_hot_targets - minus_tensor, min=0., max=1.)
            one_hot_targets += add_tensor

            return one_hot_targets

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                    frame_id: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Computes the loss

            :param inputs: the predicted logit batch. Contains x sample sequences,
                            containing y token logit predictions over the vocabulary
            :param targets: the ground truth, determined by vocab indices, in batches, for example
                            [[2, 15, 32187, ...], ...]
            :param frame_id: the frame index for each sample in batch. Expected something like [0, 3, 1, 3, ...]
            :return: the computed loss
            """
            inputs = torch.log_softmax(input=inputs, dim=-1)
            targets = self.smooth_target(inputs_type=inputs.dtype, targets=targets, frame_id=frame_id)

            return torch.mean(-torch.sum(targets * inputs, dim=-1))

    class PermuteCategoricalAccuracy:
        """
        Doesn't reward expected paddings on its own
        """

        def __init__(self, ac_tokenizer: transformers.PreTrainedTokenizer, top_k=1):
            self.ac_tokenizer = ac_tokenizer
            self.top_k = top_k
            self.correct_count = 0
            self.total_count = 0

        def __call__(self, y_pred: Tensor, y_true: Tensor):
            top_k = y_pred.topk(self.top_k)[1]
            y_true = y_true.clone()
            y_true[y_true == self.ac_tokenizer.pad_token_id] = -1
            true_k = y_true.unsqueeze(dim=-1).repeat_interleave(repeats=self.top_k, dim=-1)
            self.correct_count += top_k.eq(true_k).float().sum().item()
            self.total_count += torch.count_nonzero(y_true + 1)
            try:
                accuracy = 100. * self.correct_count / self.total_count
            except ZeroDivisionError:
                accuracy = 0
            return accuracy

        def reset(self):
            self.correct_count = 0
            self.total_count = 0

    class Training(pytorch_lightning.LightningModule):
        """
        A wrapped transformer, ready for training (imitates tensorflow.keras.Model.fit()
        """

        def __init__(self, model: transformers.PreTrainedModel,
                     t_tokenizer: transformers.PreTrainedTokenizer,
                     additional_trainings_args: Optional[Dict] = None, *args: Any, **kwargs: Any) -> None:
            if "on_gpu" in kwargs:
                on_gpu = kwargs.pop("on_gpu")
                logger.debug("\"on-gpu\" ({}) is an expected hparam loaded by the internal method "
                             "cls.load_from_checkpoint. Without removing it, we would get a "
                             "TypeError: __init__() got an unexpected keyword argument 'on_gpu'", on_gpu)
            super().__init__(*args, **kwargs)
            self.model = model
            self.t_tokenizer = t_tokenizer
            if additional_trainings_args is None:
                additional_trainings_args = dict()
            self.loss_fn = T5Trainer.CustomCrossEntropyLoss(
                vocabulary_size=len(t_tokenizer.get_vocab()),
                label_smoothing=additional_trainings_args.get("label_smoothing", .0),
                tdf_vocab_smoothing_factor=additional_trainings_args.get("tdf", None),
                dict_frame_vocab_smoothing_factor=additional_trainings_args.get("frame_words", None)
            )
            self.metrics = {"acc": T5Trainer.PermuteCategoricalAccuracy(ac_tokenizer=t_tokenizer),
                            "acc_top3": T5Trainer.PermuteCategoricalAccuracy(ac_tokenizer=t_tokenizer, top_k=3)}
            self.checkpoint = pytorch_lightning.callbacks.ModelCheckpoint(monitor="val_acc", mode="max",
                                                                          save_weights_only=True)
            self.save_hyperparameters("t_tokenizer", "additional_trainings_args")

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path: pathlib.Path,
                                 map_location=None, hparams_file=None, strict=True,
                                 raw_model: Optional[transformers.PreTrainedTokenizer] = None):
            logger.info("You want to load the checkpoint \"{}\"", checkpoint_path)

            if raw_model is None:
                logger.warning("You suggested no model (we only saved the weights) - we assume you want to load T5...")
                model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")
            else:
                model = raw_model
                logger.debug("Raw model: {}", type(model))

            return super(cls, cls).load_from_checkpoint(
                checkpoint_path=str(checkpoint_path.absolute()),
                map_location=map_location,
                hparams_file=str(checkpoint_path.parent.parent.joinpath("hparams.yaml"))
                if hparams_file is None else hparams_file,
                strict=strict,
                model=model
            )

        def forward(self, *args, **kwargs) -> Any:
            return self.model.forward(*args, **kwargs)

        def define_model_inputs(self, batch: Tensor) -> Dict:
            ret = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "decoder_input_ids": torch.constant_pad_nd(input=batch[-1],
                                                           pad=(1, 0),
                                                           value=self.t_tokenizer.pad_token_id)[:, :-1]
            }

            if isinstance(self.model, FrameBiasedT5ForConditionalGeneration):
                ret["frame_ids"] = batch[2]

            return ret

        def training_step(self, batch: Tensor, *args, **kwargs) -> STEP_OUTPUT:
            self.model.train(mode=True)

            out = self.model(**self.define_model_inputs(batch))["logits"]
            loss = self.loss_fn(out, batch[-1], batch[2]) \
                if isinstance(self.loss_fn, T5Trainer.CustomCrossEntropyLoss) else self.loss_fn(out, batch[-1])
            ret = {"loss": loss}
            self.log(name="train_loss", value=loss.item() if isinstance(loss, Tensor) else loss,
                     prog_bar=False, logger=True, on_epoch=False, on_step=True, reduce_fx=torch.min)
            acc_dict = {"train_{}".format(k): m(out, batch[-1]) for k, m in self.metrics.items()}
            for k, v in acc_dict.items():
                self.log(name=k, value=v, prog_bar=True, logger=True, on_epoch=True, reduce_fx=torch.max)
            ret.update(acc_dict)
            return ret

        def validation_step(self, batch: Tensor, *args, **kwargs) \
                -> Optional[STEP_OUTPUT]:
            self.model.eval()
            out = self.model(**self.define_model_inputs(batch))["logits"]
            loss = self.loss_fn(out, batch[-1], batch[2]) \
                if isinstance(self.loss_fn, T5Trainer.CustomCrossEntropyLoss) else self.loss_fn(out, batch[-1])
            ret = {"val_loss": loss}
            acc_dict = {"val_{}".format(k): m(out, batch[-1]) for k, m in self.metrics.items()}
            ret.update(acc_dict)
            for k, v in ret.items():
                self.log(name=k, value=v.item() if isinstance(v, Tensor) else v,
                         prog_bar=True, logger=True, on_epoch=True, reduce_fx=torch.mean)
            return ret

        def test_step(self, batch: Tensor, *args, **kwargs) \
                -> Optional[STEP_OUTPUT]:
            self.model.eval()
            out = self.model(**self.define_model_inputs(batch))["logits"]
            loss = self.loss_fn(out, batch[-1], batch[2]) \
                if isinstance(self.loss_fn, T5Trainer.CustomCrossEntropyLoss) else self.loss_fn(out, batch[-1])
            ret = {"test_loss": loss}
            acc_dict = {"test_{}".format(k): m(out, batch[-1]) for k, m in self.metrics.items()}
            ret.update(acc_dict)
            for k, v in ret.items():
                self.log(name=k, value=v.item() if isinstance(v, Tensor) else v,
                         prog_bar=False, logger=True, on_epoch=True, reduce_fx=torch.mean)
            return ret

        def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
            for metric in self.metrics.values():
                metric.reset()
            super().training_epoch_end(outputs)
            logger.success("A training epoch ends. Reset {} metrics", len(self.metrics))
            if isinstance(self.model, FrameBiasedT5ForConditionalGeneration) and not self.model.fast:
                logger.info("Weight matrices of the logits-frame-feeder: {}", self.model.lin_frame_layer.weight)

        def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
            for metric in self.metrics.values():
                metric.reset()
            super().training_epoch_end(outputs)

        def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
            for metric in self.metrics.values():
                metric.reset()
            super().training_epoch_end(outputs)

        def configure_callbacks(self):
            return [pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=2),
                    self.checkpoint]

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=2e-4, weight_decay=1e-7)
            return {"optimizer": optimizer,
                    "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=.975),
                    "interval": "step",
                    "frequency": 32
                    }

    def init_trainer(self, additional_training_args: Optional[Dict] = None):
        logger.trace("Initialize the required modules...")
        self.trainer_module = T5Trainer.Training(model=self.model, t_tokenizer=self.tokenizer,
                                                 additional_trainings_args=additional_training_args)
        additional_root_dir_paths = []
        if additional_training_args is not None:
            if "label_smoothing" in additional_training_args:
                additional_root_dir_paths.append("smoothing{}".format(additional_training_args.pop("label_smoothing")))
            if "tdf" in additional_training_args:
                additional_root_dir_paths.append("tdf{}".format(
                    round(torch.sub(1, torch.min(additional_training_args.pop("tdf"))).item(), 2)
                ))
            if "frame_words" in additional_training_args:
                additional_root_dir_paths.append("frame{}".format(len(additional_training_args.pop("frame_words"))))
        root_dir = pathlib.Path(
            ".out", "pytorch_lightning", type(self.model).__name__,
            "{}-{}".format(len(self.train_x["input_ids"][0]), len(self.train_y["input_ids"][0])),
            *additional_root_dir_paths
        )
        logger.debug("Base path: {}", root_dir.absolute())
        root_dir.mkdir(parents=True, exist_ok=True)
        self.teacher = pytorch_lightning.Trainer(check_val_every_n_epoch=1, min_epochs=2, max_epochs=12,
                                                 log_every_n_steps=8, flush_logs_every_n_steps=16,
                                                 progress_bar_refresh_rate=1,
                                                 default_root_dir=str(root_dir.absolute()),
                                                 # https://github.com/PyTorchLightning/pytorch-lightning/issues/5604
                                                 **(dict()
                                                    if isinstance(self.model, FrameBiasedT5ForConditionalGeneration) else
                                                    {"num_processes": 2,
                                                     "plugins": DDPPlugin(find_unused_parameters=True)}),
                                                 **(additional_training_args or {}))
        logger.debug("Initialize: {}, {}", self.trainer_module, self.teacher)

    def train(self) -> Optional[pathlib.Path]:
        if self.teacher is None or self.trainer_module is None:
            logger.debug("First, we have to initialize the trainer module...")
            self.init_trainer(
                additional_training_args=dict(self.additional_training_args)
                if self.additional_training_args is not None else None
            )

        assert isinstance(self.trainer_module, T5Trainer.Training)
        assert isinstance(self.teacher, pytorch_lightning.Trainer)

        self.teacher.fit(model=self.trainer_module,
                         train_dataloaders=DataLoader(dataset=TensorDataset(self.train_x["input_ids"],
                                                                            self.train_x["attention_mask"],
                                                                            self.train_x["frame_index"],
                                                                            self.train_y["input_ids"]),
                                                      batch_size=16, shuffle=True),
                         val_dataloaders=DataLoader(dataset=TensorDataset(self.val_x["input_ids"],
                                                                          self.val_x["attention_mask"],
                                                                          self.val_x["frame_index"],
                                                                          self.val_y["input_ids"]),
                                                    batch_size=24, shuffle=False))

        self.teacher.logger.save()

        logger.success("Successfully trained and logged by \"{}\" into {}: {}",
                       self.teacher.logger.name,
                       self.teacher.logger.save_dir if self.teacher.logger.save_dir is not None else "- no dict -",
                       self.teacher.logger.experiment)

        if self.teacher.logger.save_dir is not None:
            logger.info("We can't provide a proper history. Nevertheless, you can just download the logs, navigate to "
                        "{} and call \"tensorboard — logdir=./\" "
                        "(see https://towardsdatascience.com/converting-from-keras-to-pytorch-lightning-be40326d7b7d)",
                        self.teacher.logger.save_dir)

        self.teacher.checkpoint_connector.restore_model_weights(
            checkpoint_path=self.trainer_module.checkpoint.best_model_path
        )
        self.model = self.trainer_module.model
        logger.info("Restore best transformer {}", str(self.model)[:min(len(str(self.model)), 33)])

        return pathlib.Path(self.teacher.logger.save_dir, "version_{}".format(self.teacher.logger.version)) \
            if self.teacher.logger.save_dir is not None else None

    def test(self):
        if self.teacher is None or self.trainer_module is None:
            logger.debug("First, we have to initialize the trainer module...")
            self.init_trainer(
                additional_training_args=dict(self.additional_training_args)
                if self.additional_training_args is not None else None
            )

        props_test = self.teacher.test(
            model=self.trainer_module,
            dataloaders=DataLoader(dataset=TensorDataset(self.test_x["input_ids"],
                                                         self.test_x["attention_mask"],
                                                         self.test_x["frame_index"],
                                                         self.test_y["input_ids"]),
                                   batch_size=32, shuffle=False),
            ckpt_path="best",
            verbose=True
        )

        logger.success("Test finished: {}", props_test)

    def generate(self, limit=-1, min_length=2, max_length=24, cherry_picker: Optional[CherryPicker] = None) -> Dict:
        ret = {"columns": ["input", "ground_truth", "prediction_debug", "prediction"]}

        if limit >= 1 and limit > len(self.test_x["input_ids"]):
            logger.warning("You have only {} samples in your test data, but you want generate for {} sample - we're "
                           "sorry to produce {} fewer", len(self.test_x["input_ids"]), limit,
                           limit-len(self.test_x["input_ids"]))
        elif limit <= 0:
            logger.info("Generate conclusions for all samples in your test split ({}).", len(self.test_x["input_ids"]))
        else:
            logger.trace("Limit: {}", limit)

        for i, data in enumerate(
                zip(self.test_x["input_ids"][:limit] if limit >= 1 else self.test_x["input_ids"],
                    self.test_x["attention_mask"][:limit] if limit >= 1 else self.test_x["attention_mask"],
                    (self.test_x["frame_index"][:limit] if limit >= 1 else self.test_x["frame_index"])
                    if "frame_index" in self.test_x else None,
                    self.test_y["input_ids"][:limit] if limit >= 1 else self.test_y["input_ids"])):
            sample_x, sample_x_attention, sample_x_frame_id, sample_y = data

            plain_input_premise = self.tokenizer.decode(token_ids=sample_x,
                                                        skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
            plain_ground_truth = self.tokenizer.decode(token_ids=sample_y,
                                                       skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)

            outputs = self.model.generate(
                input_ids=torch.unsqueeze(sample_x, dim=0),
                attention_mask=torch.unsqueeze(sample_x_attention, dim=0),
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                num_beams=5 if cherry_picker is None else 12,
                top_k=50,
                top_p=.925,
                temperature=0.75 if cherry_picker is None else 1.1,  # higher temperature: more word diversity
                no_repeat_ngram_size=2,
                encoder_no_repeat_ngram_size=-1,
                length_penalty=1.2,
                repetition_penalty=1.25,
                return_dict_in_generate=True,
                remove_invalid_values=True,
                num_return_sequences=1 if cherry_picker is  None else 8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                output_scores=True,
                output_attentions=False,
                output_hidden_states=False,
                decoder_start_token_id=self.tokenizer.pad_token_id,
                forced_eos_token_id=None,
                **({"frame_ids": torch.unsqueeze(sample_x_frame_id, dim=0)}
                   if isinstance(self.model, FrameBiasedT5ForConditionalGeneration) else dict())
            )

            if cherry_picker is None:
                output_ids = outputs.sequences[0]

                plain_prediction = self.tokenizer.decode(token_ids=output_ids,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
                plain_prediction_debug = self.tokenizer.decode(token_ids=output_ids,
                                                               skip_special_tokens=False,
                                                               clean_up_tokenization_spaces=False)
            else:
                logger.trace("Oh, look, there is a cherry-picker: {}", cherry_picker)

                logger.debug("Select the best one out of {} prediction sequences", len(outputs.sequences))
                plain_premise_predictions = []
                plain_predictions_debug = []
                for seq in outputs.sequences:
                    plain_premise_predictions.append(
                        (plain_input_premise,
                         self.tokenizer.decode(token_ids=seq, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True))
                    )
                    plain_predictions_debug.append(
                        self.tokenizer.decode(token_ids=seq, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
                    )
                logger.trace("Collected {} predictions: {}", len(plain_premise_predictions),
                             " +++ ".join(map(lambda p: p[-1], plain_premise_predictions)))
                plain_prediction, pos = \
                    cherry_picker.cherry_picking(generated_sequences=plain_premise_predictions,
                                                 reference=plain_ground_truth)
                plain_prediction_debug = plain_predictions_debug[pos]

            logger.debug("Predicting \"{}\" --> \"{}\"", plain_prediction_debug, plain_prediction)
            if plain_prediction == plain_ground_truth:
                logger.success("We predict the ground truth \"{}\" -> \"{}\"", plain_input_premise, plain_ground_truth)
            else:
                logger.warning("\"{}\": Should be \"{}\", but is \"{}\"", plain_input_premise, plain_ground_truth,
                               plain_prediction)

            ret["test_{}".format(i)] = {
                "input": plain_input_premise,
                "ground_truth": plain_ground_truth,
                "prediction_debug": plain_prediction_debug,
                "prediction": plain_prediction
            }

        return ret
