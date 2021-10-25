import os

import torch
from torch import Tensor, nn
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import Linear, Sigmoid, Softmax
from typing import Dict, Optional, Union, Any, Tuple

from loguru import logger


def post_process_out(out: Seq2SeqLMOutput, return_dict: bool):
    if return_dict is not None and not return_dict:
        output = (out.logits,) + out.past_key_values + out.decoder_hidden_states + out.decoder_attentions + \
                 out.encoder_last_hidden_state + out.encoder_hidden_states + out.encoder_attentions
        return ((out.loss,) + output) if out.loss is not None else output
    else:
        return out


class FrameBiasedT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config, frame_dict: Dict[int, Tensor], fast: bool = True, sequence_length: Optional[int] = None):
        super().__init__(config)
        if torch.cuda.device_count() == 1:
            logger.info("You're using the (1) GPU - hence we must assign all {} tensors of the frame-dict to the GPU.",
                        len(frame_dict))
            self.frame_dict = {k: tensor.cuda() for k, tensor in frame_dict.items()}
        logger.info("Received a frame_dict, containing {} frames", len(frame_dict))
        if -1 not in frame_dict:
            logger.warning("There is no default (fallback) frame (-1) in the frame_dict included, only {}",
                           ", ".join(map(lambda k: str(k), frame_dict.keys())))
        self.fast = fast
        if not fast:
            self.lin_frame_layer = Linear(in_features=super().get_output_embeddings().out_features,
                                          out_features=config.vocab_size,
                                          bias=False)
            self.sig_frame_layer = Sigmoid()
        else:
            self.softmax_frame_layer = Softmax(dim=-1)

        self.sequence_length = sequence_length
        self.frame_dict_sequence = None
        if self.sequence_length is not None:
            logger.info("Preprocess the frame weights (to a sequence length of {})", sequence_length)
            self.frame_dict_sequence = {k: v.repeat(sequence_length, 1) for k, v in self.frame_dict.items()}
        logger.success("Successfully initialized {}{}", self, " (fast)" if fast else "")

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None,
                past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, frame_ids=None):
        out = super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask,
                              decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds,
                              decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states,
                              True)
        assert isinstance(out, Seq2SeqLMOutput), "The expected return type of the Transformer is not met. " \
                                                 "Instead, it is {}".format(type(out))

        logger.trace("Calculated the raw outputs: {}", out)
        logger.trace("Shape of the after-processing logits: {}", out.logits.shape)

        #logger.info("Let's start the frame-correction (with frame-ids {})", frame_ids)

        if frame_ids is None:
            logger.warning("You didn't provide any frame-id, hence, this module will behave like a normal "
                           "T5ForConditionalGeneration")
            return post_process_out(out=out, return_dict=return_dict)
        else:
            def move_to_same_device(tensor: Tensor) -> Tensor:
                if torch.cuda.device_count() > 1:
                    logger.debug("You're using {} GPUs - hence, we must ensure that all tensors are in the same device!",
                                 torch.cuda.device_count())
                    return tensor.to(out.logits.device)
                return tensor

            logger.trace("Fetched following frame-ids: {}", frame_ids)
            post_scaled_lm = 0.5 * self.softmax_frame_layer(out.logits) if self.fast else \
                self.sig_frame_layer(self.lin_frame_layer(out.logits))
            if isinstance(frame_ids, int):
                multiplication = move_to_same_device(self.frame_dict.get(frame_ids, self.frame_dict.get(-1)))
                if len(post_scaled_lm.shape) > 1:
                    multiplication = multiplication.repeat(*post_scaled_lm.shape[:-1], 1)
            elif isinstance(frame_ids, torch.Tensor):
                if frame_ids.shape.numel() == 1:
                    multiplication = move_to_same_device(self.frame_dict.get(frame_ids.item(), self.frame_dict.get(-1)))
                    if len(post_scaled_lm.shape) > 1:
                        multiplication = multiplication.repeat(*post_scaled_lm.shape[:-1], 1)
                else:
                    assert len(post_scaled_lm) == len(frame_ids), \
                        "The batch sizes are not equal: T5_out({}) vs. frame_ids({})".format(post_scaled_lm.shape,
                                                                                             frame_ids.shape)

                    if self.sequence_length is not None and post_scaled_lm.shape[1] == self.sequence_length:
                        multiplication = \
                            move_to_same_device(
                                torch.stack(
                                    [self.frame_dict_sequence.get(i.item(), self.frame_dict_sequence.get(-1))
                                     for i in frame_ids]
                                )
                            )
                    else:
                        if self.sequence_length is not None:
                            logger.warning("You configured a sequence-length-performance-boost, but you can't use it. "
                                           "Expected a sequence length of {}, but shape is {}", self.sequence_length,
                                           post_scaled_lm.shape)
                        multiplication = \
                            move_to_same_device(
                                torch.stack(
                                    [self.frame_dict.get(i.item(),
                                                         self.frame_dict.get(-1)).repeat(*post_scaled_lm.shape[1:-1], 1)
                                     for i in frame_ids]
                                )
                            )
                    logger.trace("Successfully stacked the frame multiplication tensor: {} (out of frames: {})",
                                 multiplication.shape, frame_ids)
            else:
                logger.error("frame_ids are of unexpected type \"{}\" - ignore the frame-bias!", type(frame_ids))
                multiplication = .5

            post_scaled_lm = post_scaled_lm * multiplication
            logger.trace("The post-frame-scaled outputs (frame {}) are: {}", frame_ids, post_scaled_lm)

            out.logits = out.logits * (.5 + post_scaled_lm)

            return post_process_out(out=out, return_dict=return_dict)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, head_mask=None,
                                      decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None,
                                      encoder_outputs=None, **kwargs):
        input_dict = super().prepare_inputs_for_generation(input_ids, past, attention_mask, head_mask,
                                                           decoder_head_mask, cross_attn_head_mask, use_cache,
                                                           encoder_outputs, **kwargs)

        if "frame_ids" in kwargs and "frame_ids" not in input_dict:
            input_dict["frame_ids"] = kwargs["frame_ids"]
        else:
            logger.warning("Please provide an additional \"frame_ids\"-param in the .generate-method!")

        return input_dict

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs) \
            -> Dict[str, Any]:
        encoder_kwargs = dict(model_kwargs)
        frame_ids = None
        if "frame_ids" in encoder_kwargs:
            frame_ids = encoder_kwargs.pop("frame_ids")
            logger.debug("The encoder shouldn't get the frame-ids (=> FAIL), so lets skip: {}", frame_ids)
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(input_ids, encoder_kwargs)
        if frame_ids is not None:
            model_kwargs["frame_ids"] = frame_ids

        return model_kwargs

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        embedding = super().resize_token_embeddings(new_num_tokens)

        if new_num_tokens is None:
            logger.error("Can't resize the vocab without a valid number")
            if not self.fast:
                self.lin_frame_layer = Linear(in_features=super().config.vocab_size,
                                              out_features=super().config.vocab_size,
                                              bias=False)
        elif not self.fast:
            if super().get_output_embeddings().out_features != self.lin_frame_layer.out_features:
                logger.debug("Change the features of the linear frame layer to {}",
                             super().get_output_embeddings().out_features)
                self.lin_frame_layer = Linear(in_features=super().get_output_embeddings().out_features,
                                              out_features=super().get_output_embeddings().out_features,
                                              bias=False)
            else:
                logger.trace("No additional adaptions necessary...")

        return embedding

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError("T5 does not support more/ less positions (length)")

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise NotImplementedError("T5 does not support more/ less positions (length)")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if "frame_dict" not in kwargs:
            logger.warning("Please provide the variable \"frame_dict\" (Dict[int, Tensor])")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)



