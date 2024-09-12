import logging
from dataclasses import dataclass

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import SequenceSummary
from transformers.models.bart import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartClassificationHead


def get_gen_model_init(
    transformer: str,
    use_cl: bool,
    tokenizer: PreTrainedTokenizer = None,
    pretrained: str = None,
    freeze_encoder: bool = False,
    freeze_encoder_embedding: bool = False,
    lambda_pretrained_embedding_weight_decay: float = 0.0,
    lambda_pretrained_output_weight_decay: float = 0.0,
    train_layers: int = -1,
    reinit_layers: int = -1,
    output_layers=None,
    adapter_mode=None,
    **init_kwargs,
):
    init_kwargs = init_kwargs.copy()
    init_kwargs.update(
        {
            "pretrained_model_name_or_path": (
                pretrained if pretrained is not None else transformer
            ),
            "local_files_only": True if pretrained is not None else False,
        }
    )

    def wrapper(pretrained=None, **additional_kwargs):
        if pretrained is not None:
            init_kwargs.update(
                {"pretrained_model_name_or_path": pretrained, "local_files_only": True}
            )

        def wrapper2():
            # return_dict=True -> forward call returns object
            if "bart" in transformer:
                if use_cl:
                    model = (
                        BartForConditonalGenerationWithBinaryClassifier.from_pretrained(
                            **init_kwargs, **additional_kwargs, num_labels=2
                        )
                    )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        **init_kwargs, **additional_kwargs, return_dict=True
                    )
            else:
                if use_cl:
                    raise NotImplementedError(
                        "Contrastive Loss hasn't been implemented for other models than bart"
                    )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    **init_kwargs, **additional_kwargs, return_dict=True
                )

            # if the model is loaded from a checkpoint then we don't have to resize embeddings
            if (
                pretrained is None
                and tokenizer is not None
                and len(tokenizer) != model.get_input_embeddings().weight.size(dim=0)
            ):
                logging.warning(
                    f"Resizing embedding layer from {model.get_input_embeddings().weight.size(dim=0)} to {len(tokenizer)}, make sure that you train the model or load compatible weights."
                )
                model.resize_token_embeddings(len(tokenizer))
            return model

        return wrapper2

    return wrapper


class BartForConditonalGenerationWithBinaryClassifier(BartForConditionalGeneration):
    def __init__(
        self, config: BartConfig, cl_inner_dim: int, cl_dropout: float = 0.0, **kwargs
    ):
        super().__init__(config, **kwargs)
        config.num_labels = 2
        # we only use the summary and add our own projection afterwards
        config.summary_type = "cls_index"
        config.summary_use_proj = False
        self.summary = SequenceSummary(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            cl_inner_dim,
            config.num_labels,
            cl_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        cl_labels=None,
        cl_token_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            assert decoder_input_ids is not None

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # hidden state of last layer of decoder
        # hidden_states_enc = outputs[-1]  # hidden state last layer of encoder

        lm_logits = self.lm_head(hidden_states) + self.final_logits_bias
        cl_logits = None
        if decoder_input_ids is not None:
            cl_logits = self.classification_head(
                self.summary(hidden_states, cl_token_ids)
            )

        loss = None
        if labels is not None or cl_labels is not None:
            loss_fct = CrossEntropyLoss()
            if labels is not None:
                loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
                )
            if cl_labels is not None:
                cl_loss = loss_fct(
                    cl_logits.view(-1, self.config.num_labels), cl_labels.view(-1)
                )
                if loss is None:
                    loss = cl_loss
                else:
                    loss += cl_loss

        if not return_dict:
            output = (lm_logits, cl_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMWithBinaryOutput(
            loss=loss,
            logits=lm_logits,
            cl_logits=cl_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@dataclass
class Seq2SeqLMWithBinaryOutput(Seq2SeqLMOutput):
    cl_logits: torch.FloatTensor = None


@dataclass
class DataCollatorForSeq2SeqWithBinaryClassifier(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    def __call__(self, features, **kwargs):
        decoder_input_ids = (
            [feature["decoder_input_ids"] for feature in features]
            if "decoder_input_ids" in features[0].keys()
            else None
        )
        # We have to pad the decoder_input_ids before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if decoder_input_ids is not None:
            max_label_length = max(len(l) for l in decoder_input_ids)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_label_length - len(feature["decoder_input_ids"])
                )
                feature["decoder_input_ids"] = (
                    feature["decoder_input_ids"] + remainder
                    if padding_side == "right"
                    else remainder + feature["decoder_input_ids"]
                )

        return super().__call__(features, **kwargs)
