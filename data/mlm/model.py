import logging

from transformers import AutoModelForMaskedLM, PreTrainedTokenizer


def get_mlm_model_init(
    transformer: str,
    tokenizer: PreTrainedTokenizer = None,
    pretrained: str = None,
    lambda_pretrained_embedding_weight_decay: float = 0.0,
    lambda_pretrained_output_weight_decay: float = None,
    train_layers: int = -1,
    reinit_layers: int = -1,
    freeze_encoder: bool = None,
    freeze_encoder_embedding: bool = None,
    output_layers=None,
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
            model = AutoModelForMaskedLM.from_pretrained(
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
