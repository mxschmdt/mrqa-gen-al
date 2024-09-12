# delay evaluation of annotation
from __future__ import annotations

import logging
from typing import List, Tuple, Union

import torch
from opendelta import AdapterModel, BitFitModel, LoraModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import AutoModel

from ..utils.model import SaveLoadModelMixin


def get_rc_model_init(**kwargs):
    def wrapper(**additional_kwargs):
        # transformers expects 0 or 1 arguments for model init function
        def wrapper2():
            return QAModel(**kwargs, **additional_kwargs)

        return wrapper2

    return wrapper


class QAModel(torch.nn.Module, SaveLoadModelMixin):
    def __init__(
        self,
        transformer: str,
        cache_dir: Union[None, str] = None,
        output_layers: List[int] = None,
        freeze_encoder: bool = False,
        freeze_encoder_embedding: bool = False,
        pretrained: str = None,
        lambda_pretrained_embedding_weight_decay: float = 0.0,
        lambda_pretrained_output_weight_decay: float = 0.0,
        train_layers: int = -1,
        reinit_layers: int = -1,
        output_hidden_states: bool = None,
        adapter_mode: str = None,
        **kwargs,
    ):
        # kwargs will collect arguments unnecessary for this class
        super().__init__()
        output_layers = output_layers + [2] if output_layers is not None else [2]
        self.loss_fn = CrossEntropyLoss()
        self.output_hidden_states = output_hidden_states

        self.embedder = AutoModel.from_pretrained(
            transformer,
            cache_dir=cache_dir,
            return_dict=False,
            output_hidden_states=output_hidden_states,
        )  # return_dict=False -> forward call returns tuple

        logging.info(f"Training of encoder weights: {not freeze_encoder}")
        logging.info(
            f"Training of encoder embedding weights: {not (freeze_encoder or freeze_encoder_embedding)}"
        )
        if freeze_encoder:
            # disable gradient computation for all parameters of the embedder (disables training on this part)
            for param in self.embedder.parameters():
                param.requires_grad = False
        elif freeze_encoder_embedding:
            # if encoder is frozen then embedding is frozen anyway
            # freeze encoder embedding
            for name, param in self.embedder.named_parameters():
                if name.startswith("embeddings."):
                    param.requires_grad = False

        num_embedding_layers = len(self.embedder.encoder.layer)
        if train_layers >= 0:
            # only train layers indexed from top, i.e. `train_layers==0` => only output layer, `train_layers==1` => output layer + last layer from embedder, ...
            # that means embedding weights are not trained as well
            logging.info(
                f"Train only weights from output layer{f' and top {train_layers} layers of the encoder' if train_layers >= 1 else ''}."
            )
            # first disable gradient computation (and hence updates) for all parameters of the embedder
            # NOTE will also disable embedding weight gradient computation
            for param in self.embedder.parameters():
                param.requires_grad = False
            if train_layers >= 1:
                # enable gradient computation for specified layers of the embedder
                for name, param in self.embedder.named_parameters():
                    # we look for parameters with name "encoder.layer.i" where `i` is index of the layer counting from bottom
                    if name.startswith("encoder.layer."):
                        if train_layers >= (
                            num_embedding_layers - int(name.split(".")[2])
                        ):
                            param.requires_grad = True

        # set up output layers
        qa_outputs_list = []
        input_dim = self.embedder.config.hidden_size
        for output_dim in output_layers:
            if qa_outputs_list:
                # starting from the second output layer we add a ReLu first
                qa_outputs_list.append(torch.nn.ReLU())
            qa_outputs_list.append(torch.nn.Linear(input_dim, output_dim, bias=True))
            input_dim = output_dim
        # self.qa_outputs = torch.nn.Sequential(*qa_outputs_list)
        # NOTE the following single linear layer instead of a sequential module is needed if we use weights from old models
        self.qa_outputs = torch.nn.Linear(
            self.embedder.config.hidden_size, 2, bias=True
        )
        self.qa_outputs.apply(self.embedder._init_weights)
        logging.info(f"Head architecture: {self.qa_outputs}")

        if pretrained is not None:
            # load pre-trained weights
            self.load(pretrained)

        if reinit_layers >= 0:
            # re-initialize layers indexed from top
            logging.info(
                f"Re-initialize weights from output layer{f' and top {reinit_layers} layers of the encoder' if reinit_layers >= 1 else ''}."
            )
            # re-initialize output layer (layer 0)
            self.qa_outputs.apply(self.embedder._init_weights)
            if reinit_layers >= 1:
                # re-initialize specified layer of the embedder
                # start with the last layer
                for layer in self.embedder.encoder.layer[-1 : -1 - reinit_layers : -1]:
                    layer.apply(self.embedder._init_weights)

        if adapter_mode is not None and adapter_mode not in ["none", "None", "full"]:
            # NOTE current naming is for Roberta
            # apply adapters (using OpenDelta)
            if adapter_mode == "simple":
                # simple will add weights to all feed-forward layers
                delta_model = AdapterModel(
                    backbone_model=self.embedder,
                    modified_modules=["[r]encoder.*.dense"],
                    bottleneck_dim=12,
                )
                delta_model.freeze_module(
                    exclude=["deltas", "layernorm_embedding"], set_state_dict=True
                )
                delta_model.log()
            elif adapter_mode == "lora":
                delta_model = LoraModel(
                    backbone_model=self.embedder,
                    modified_modules=["attention.self.key", "attention.self.value"],
                )
                delta_model.freeze_module(
                    exclude=["deltas", "layernorm_embedding"], set_state_dict=True
                )
                delta_model.log()
            elif adapter_mode == "bitfit":
                delta_model = BitFitModel(
                    backbone_model=self.embedder,
                    modified_modules=["attention", "dense", "LayerNorm"],
                )
                delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
                delta_model.log()
            else:
                raise ValueError(
                    f"'{adapter_mode}' is not a valid option for adapter mode"
                )
            logging.info(f"Using adapter mode '{adapter_mode}'")
        else:
            logging.info("Adapters disabled")

        # store the pre-trained weights for use in weight decay as a dictionary
        logging.info(
            f"Using pre-trained embedding weight decay: {lambda_pretrained_embedding_weight_decay if lambda_pretrained_embedding_weight_decay != .0 else 'Disabled'}"
        )
        self.lambda_pretrained_embedding_weight_decay = (
            lambda_pretrained_embedding_weight_decay
        )
        if lambda_pretrained_embedding_weight_decay != 0.0:
            # pre-trained embedding weight regularization
            # we don't know on which device the modell wil be during training hence we do the regularization computation always on the cpu
            self.embedding_init_weights = {
                name: param.data.clone().cpu()
                for name, param in self.embedder.named_parameters()
            }
        logging.info(
            f"Using pre-trained output layer weight decay: {lambda_pretrained_output_weight_decay if lambda_pretrained_output_weight_decay != .0 else 'Disabled'}"
        )
        self.lambda_pretrained_output_weight_decay = (
            lambda_pretrained_output_weight_decay
        )
        if lambda_pretrained_output_weight_decay != 0.0:
            # pre-trained output layer weight regularization
            # we don't know on which device the model will be during training hence we do the regularization computation always on the cpu
            self.output_init_weights = {
                name: param.data.clone().cpu()
                for name, param in self.qa_outputs.named_parameters()
            }

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        transformers_fix: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # input is batch x sequence
        # some models do not need token_type_ids (or it is created on the fly but doesn't need special processing)

        outputs = self.embedder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        embedding = outputs[0]
        logits: torch.Tensor
        logits = self.qa_outputs(embedding)

        # set padded values in output to small value
        mask = (1 - attention_mask).bool().unsqueeze(2).expand(-1, -1, 2)
        logits = logits.masked_fill(mask, float("-inf"))

        probs = logits.softmax(dim=1)
        # for some reason transformers gets rid of first dimension if batch size is one leading to a concatenation of sequences instead of samples/batches
        # hence we add an additional dimension so that this one is removed
        # NOTE check for len(probs) first since transformers_fix will be a tensor rather than a scalar if batch size > 1
        if len(probs) == 1 and probs.dim() == 3 and transformers_fix:
            probs = probs.unsqueeze(0)

        if labels is not None:
            # compute loss
            loss = self.loss_fn(logits, labels)  # mean over start and end loss
            if self.lambda_pretrained_embedding_weight_decay != 0.0:
                # add pre-trained embedding weight decay (l2)
                norm = 0.0
                for name, param in self.embedder.named_parameters():
                    norm += (
                        torch.norm(param.data.cpu() - self.embedding_init_weights[name])
                        ** 2
                    )
                loss += self.lambda_pretrained_embedding_weight_decay / 2 * norm
            if self.lambda_pretrained_output_weight_decay != 0.0:
                # add pre-trained output layer weight decay (l2)
                norm = 0.0
                for name, param in self.qa_outputs.named_parameters():
                    norm += (
                        torch.norm(param.data.cpu() - self.output_init_weights[name])
                        ** 2
                    )
                loss += self.lambda_pretrained_output_weight_decay / 2 * norm
            if self.output_hidden_states:
                return loss, probs, outputs[2]
            else:
                return loss, probs

        if self.output_hidden_states:
            return probs, outputs[2]
        else:
            return probs
