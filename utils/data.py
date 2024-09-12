import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Union

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    IntervalStrategy,
)
from transformers import TrainingArguments as HfTrainingArguments

from data.gen.data import (
    augment_with_negative_samples,
    prepare_ap_features,
    prepare_aqg_features,
    prepare_qg_features,
)
from data.gen.model import (
    DataCollatorForSeq2SeqWithBinaryClassifier,
    get_gen_model_init,
)
from data.mlm.data import mask_tokens_randomly, prepare_mlm_features
from data.mlm.model import get_mlm_model_init
from data.rc.data import prepare_rc_features, rectify_rc_data
from data.rc.model import get_rc_model_init
from data.utils.data import process_hf_dataset
from data.utils.utils import DataCollatorList, ProcessDataDataCollator, select_unique


@dataclass
class TrainingArguments(HfTrainingArguments):
    """`transformers.TrainingArguments` with updated default values."""

    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    per_device_eval_batch_size: int = field(
        default=40, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=2,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )

    learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate for AdamW."}
    )

    evaluation_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )

    # save_strategy: IntervalStrategy = field(
    #     default="steps",
    #     metadata={"help": "The evaluation strategy to use."},
    # )

    logging_first_step: bool = field(
        default=False, metadata={"help": "Log the first global_step"}
    )
    logging_steps: int = field(
        default=1000, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=1000, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=1000, metadata={"help": "Save checkpoint every X updates steps."}
    )

    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir (but never the best model if tracked (`load_best_model_at_end=True`)). Default is 1 checkpoint"
            )
        },
    )

    seed: Union[int, None] = field(
        default=None,
        metadata={
            "help": "Random seed that will be set at the beginning of training.",
            "nargs": "+",
        },
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )

    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to use to compare two different models."},
    )

    # report_to: Optional[List[str]] = field(
    #     default_factory=partial(list, ['comet_ml']), metadata={"help": "The list of integrations to report the results and logs to."}
    # )

    use_legacy_prediction_loop: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the legacy prediction_loop in the Trainer."
        },
    )

    tags: List[str] = field(
        default_factory=list,
        metadata={"help": "Tags to be added to the comet.ml experiment."},
    )

    # resume_from_checkpoint: Union[Optional[str,bool]] = field(
    #     default=False,
    #     metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    # )


@dataclass
class ModelArguments:
    """model arguments (passed to `init`)"""

    # NOTE better infer seq2seq and token_type_ids from model?
    # seq2seq: str2bool = field(
    #     metadata={
    #         "nargs": "?",
    #         "const": True,
    #         "help": "Whether the model is a sequence-to-sequence model or not (labels are preprocessed differently)."
    #         },
    # )

    # token_type_ids: str2bool = field(
    #     metadata={
    #         "nargs": "?",
    #         "const": True,
    #         "help": "Whether the model needs token_type_ids as input and whether to create them on pre-processing."
    #         },
    # )

    transformer: str = field(
        metadata={"help": "The transformer used in the model."},
    )

    pretrained: Optional[str] = field(
        default=None, metadata={"help": "Start with the given model checkpoint."}
    )

    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the encoder, i.e. do not train its weights."},
    )

    freeze_encoder_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Freeze the encoder's embedding weights, i.e. do not train these weights."
        },
    )

    lambda_pretrained_embedding_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained embedding weight decay lambda."}
    )

    lambda_pretrained_output_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained output layer weight decay lambda."}
    )

    train_layers: int = field(
        default=-1,
        metadata={
            "help": "Train only top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )

    reinit_layers: int = field(
        default=-1,
        metadata={
            "help": "Re-initialize top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )

    output_layers: List[int] = field(
        default=None,
        metadata={
            "help": "The layers for the span extraction head (specified as output dimensions), a layer mapping to 2 (num classes) is always added."
        },
    )

    adapter_mode: Optional[str] = field(
        default=None, metadata={"help": "The adapter mode for the model."}
    )


@dataclass
class ActiveLearningModelArguments:
    """model arguments (passed to `init`)"""

    rc_transformer: str = field(
        default=None,
        metadata={"help": "The transformer used in the rc model."},
    )

    gen_transformer: str = field(
        default=None,
        metadata={"help": "The transformer used in the gen model."},
    )

    rc_pretrained: Optional[str] = field(
        default=None, metadata={"help": "Start with the given rc model checkpoint."}
    )

    gen_pretrained: Optional[str] = field(
        default=None, metadata={"help": "Start with the given gen model checkpoint."}
    )

    rc_freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the encoder, i.e. do not train its weights."},
    )

    rc_freeze_encoder_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Freeze the encoder's embedding weights, i.e. do not train these weights."
        },
    )

    rc_lambda_pretrained_embedding_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained embedding weight decay lambda."}
    )

    rc_lambda_pretrained_output_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained output layer weight decay lambda."}
    )

    rc_train_layers: int = field(
        default=-1,
        metadata={
            "help": "Train only top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )

    rc_reinit_layers: int = field(
        default=-1,
        metadata={
            "help": "Re-initialize top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )


@dataclass
class ActiveLearningArguments:
    """Arguments for Active Learning"""

    mode: str = field(
        metadata={"help": "The query strategy for active learning."},
    )

    samples: int = field(
        metadata={"help": "The amount of samples drawn in each iteration."}
    )

    rounds: int = field(
        metadata={
            "help": "The number of times samples are drawn and the model is trained."
        }
    )


@dataclass
class DataArguments:
    """Arguments concerning data processing"""

    preprocess: bool = field(
        metadata={"help": "Will force to preprocess any data."},
    )

    rectify_answers: bool = field(
        default=False,
        metadata={"help": "Rectify answers."},
    )

    rectify_questions: bool = field(
        default=True,
        metadata={"help": "Don't rectify questions."},
    )

    eval_style: str = field(
        default="squad",
        metadata={"help": "Set the evaluation style."},
    )

    disable_no_answer: bool = field(
        default=False,
        metadata={"help": "Non-answers disabled if set to True."},
    )

    max_input_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum input length for the model, overflowing tokens will be sliced into chunks."
        },
    )

    skip_question_length: int = field(
        default=-1,
        metadata={
            "help": "'The maximum question length in tokens (questions with more tokens will be skipped)."
        },
    )

    truncate_question_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum question length in tokens (questions with more tokens will be truncated)."
        },
    )

    max_answer_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum length of the answer to predict in case of the rc model."
        },
    )

    stride: int = field(
        default=128,
        metadata={
            "help": "The maximum input length for the model, overflowing tokens will be sliced into chunks."
        },
    )

    num_worker: int = field(
        default=1,
        metadata={"help": "The number of worker used for preprocessing data."},
    )

    separate_answers: bool = field(
        default=False,
        metadata={
            "help": "Whether answers are unpacked by creating new instances or not."
        },
    )

    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Will keep preprocessed data in memory if True."},
    )

    unique: str = field(
        default=None,
        metadata={
            "metavar": "column",
            "help": "The column to be unique in the dataset. If None then no filtering is applied.",
        },
    )


def setup_model_and_data_processing(
    model_args: ModelArguments,
    task,
    cache_dir,
    stride,
    max_answer_length,
    truncate_question_length,
    max_input_length: int = None,
    num_worker: int = 1,
    preprocess: bool = False,
    keep_in_memory: bool = False,
    unique: str = None,
    skip_question_length: int = None,
    rectify_answers: bool = False,
    rectify_questions: bool = False,
    seq2seq=None,
    token_type_ids=None,
    use_cl: bool = False,
    cl_inner_dim: int = None,
    skip_retrieval: bool = False,
    eval_skip_retrieval: bool = False,
    keep_columns: str = None,
    separate_answers: bool = False,
    transformer: str = None,
):
    if model_args is None:
        assert transformer is not None
        model_args = {"transformer": transformer}
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args["transformer"], cache_dir=cache_dir, use_fast=True
    )

    # set task specific variables
    if task == "mlm":
        # mlm modeling
        model_init = get_mlm_model_init(cache_dir=cache_dir, **model_args)
        preprocess_fn = prepare_mlm_features
        preprocess_fn_kwargs = {}
        data_collator = DataCollatorList(
            (
                ProcessDataDataCollator(
                    partial(mask_tokens_randomly, tokenizer=tokenizer)
                ),
                DataCollatorForTokenClassification(tokenizer),
            )
        )  # makes sure that labels are padded
        special_token_id_map = {}
    elif task == "rc":
        # rc using question + context
        logging.info(f"Max answer length set to {max_answer_length}")
        model_init = get_rc_model_init(cache_dir=cache_dir, **model_args)
        preprocess_fn = prepare_rc_features
        preprocess_fn_kwargs = dict(question_max_length=truncate_question_length)
        data_collator = None
        special_token_id_map = {}
    elif task == "ap":
        # answer prediction/generation given context
        logging.info(f"Max answer length set to {max_answer_length}")
        model_init = get_rc_model_init(cache_dir=cache_dir, **model_args)
        preprocess_fn = prepare_ap_features
        preprocess_fn_kwargs = {}
        data_collator = None
        special_token_id_map = {}
    else:
        if not any(_model in model_args["transformer"] for _model in ["gpt2", "bart"]):
            raise ValueError(
                "Please specify a GPT2 or Bart model for question generation using --transformer"
            )
        if seq2seq is None:
            raise ValueError("You have to specify --seq2seq")
        if token_type_ids is None:
            raise ValueError("You have to specify --token_type_ids")

        bos_token_2 = None
        eos_token_2 = None
        if task == "qg":
            # question generation given context + answer
            preprocess_fn = prepare_qg_features
            preprocess_fn_kwargs = dict(
                seq2seq=seq2seq, custom_token_type_ids=token_type_ids
            )
            special_tokens_dict = None
            bos_token = "<q>"
            eos_token = tokenizer.eos_token
        elif task == "q":
            # generate question given context + anxwer
            preprocess_fn = prepare_aqg_features
            preprocess_fn_kwargs = dict(
                config="q",
                seq2seq=seq2seq,
                custom_token_type_ids=token_type_ids,
                max_question_length=truncate_question_length,
            )
            # add special tokens such that those are not split
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<a>", "<q>", "</q>"]
            }
            bos_token = "<q>"
            eos_token = "</q>"
        elif task == "aq":
            # generate answer followed by question in single decoding step given context
            preprocess_fn = prepare_aqg_features
            preprocess_fn_kwargs = dict(
                config="aq",
                seq2seq=seq2seq,
                custom_token_type_ids=token_type_ids,
                max_question_length=truncate_question_length,
            )
            # add special tokens such that those are not split
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<a>", "<q>", "</q>"]
            }
            bos_token = "<a>"
            eos_token = "</q>"
        elif task == "qa":
            # generate question followed by answer in single decoding step given context
            preprocess_fn = prepare_aqg_features
            preprocess_fn_kwargs = dict(
                config="qa",
                seq2seq=seq2seq,
                custom_token_type_ids=token_type_ids,
                max_question_length=truncate_question_length,
            )
            # add special tokens such that those are not split
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<q>", "<a>", "</a>"]
            }
            bos_token = "<q>"
            eos_token = "</a>"
        elif task == "qa2s":
            # generate question followed by answer in two deconding steps: first step uses context only, second step uses context + question
            preprocess_fn = prepare_aqg_features
            preprocess_fn_kwargs = dict(
                config="qa2s",
                seq2seq=seq2seq,
                custom_token_type_ids=token_type_ids,
                max_question_length=truncate_question_length,
            )
            # add special tokens such that those are not split
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<q>", "</q>", "<a>", "</a>"]
            }
            bos_token = "<q>"
            bos_token_2 = "<a>"
            eos_token = "</q>"
            eos_token_2 = "</a>"

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer._pad_token_type_id = 2  # pad with question type embedding

        if special_tokens_dict is not None:
            tokenizer.add_special_tokens(special_tokens_dict)

        # convert bos and eos token
        special_token_id_map = {
            "bos_token_id": tokenizer.convert_tokens_to_ids(bos_token),
            "eos_token_id": tokenizer.convert_tokens_to_ids(eos_token),
            "bos_token_id_2": (
                tokenizer.convert_tokens_to_ids(bos_token_2)
                if bos_token_2 is not None
                else None
            ),
            "eos_token_id_2": (
                tokenizer.convert_tokens_to_ids(eos_token_2)
                if eos_token_2 is not None
                else None
            ),
        }

        model_init_kwargs = dict(
            cache_dir=cache_dir, use_cl=use_cl, tokenizer=tokenizer, **model_args
        )
        if use_cl:
            # cl models need additional arguments
            model_init_kwargs["cl_inner_dim"] = cl_inner_dim
        model_init = get_gen_model_init(**model_init_kwargs)

        if seq2seq:
            # model = model_class(cache_dir=args.cache_dir, model=model_args.transformer, tokenizer=tokenizer).model
            data_collator = DataCollatorForSeq2SeqWithBinaryClassifier(
                tokenizer, model=None, label_pad_token_id=-100
            )
        else:
            data_collator = DataCollatorForTokenClassification(
                tokenizer
            )  # makes sure that labels are padded

    def preprocess_data(data: Union[Dataset, DatasetDict], is_eval_data: bool = False):
        if keep_columns is not None:
            if isinstance(data, DatasetDict):
                # we assume that every split of the dataset dict has the same columns
                _columns = set(next(iter(data.values())).column_names)
            else:
                _columns = set(data.column_names)
            data = data.remove_columns(_columns.difference(keep_columns))

            data.reset_format()

        if unique is not None:
            # select unique from specified column
            data = select_unique(data, unique, 21)

        if (
            skip_question_length is not None
            and skip_question_length != -1
            and not is_eval_data
        ):
            logging.info(
                f"Skipping training samples with question length (in tokens) > {skip_question_length}"
            )
            # we can only skip samples in the training data
            data = data.filter(
                lambda x: len(
                    tokenizer.tokenize(x["question"], add_special_tokens=False)
                )
                <= skip_question_length,
                keep_in_memory=keep_in_memory,
                num_proc=num_worker,
            )

        logging.info(
            f"{'Training' if not is_eval_data else 'Evaluation'} dataset: {data}"
        )

        if rectify_answers or rectify_questions:
            logging.info("Rectifying data")
            # TODO make sure that rectify_answers works with a list of lists
            data = process_hf_dataset(
                data,
                rectify_rc_data,
                answer_column=True if rectify_answers else None,
                question_column=True if rectify_questions else None,
                force_preprocess=preprocess,
                keep_in_memory=keep_in_memory,
                num_processes=num_worker,
            )

        def add_negative_samples(data):
            logging.info("Augmenting data with negative samples")
            unique_questions = data.unique("question")
            # answers is a dict of lists hence we have to extract the answers manually
            unique_answers = set(
                answer
                for answers in data.map(
                    lambda x: {"answers": x["answers"]["text"]},
                    remove_columns=data.column_names,
                    num_proc=num_worker,
                    load_from_cache_file=not preprocess,
                )["answers"]
                for answer in answers
            )
            # sort answers because order of set is non-deterministic (due to PYTHONHASHSEED not being set)
            unique_answers = sorted(unique_answers)
            fn_kwargs = {"questions": unique_questions, "answers": unique_answers}
            data = data.map(
                augment_with_negative_samples,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=num_worker,
            )
            logging.info(f"Augmented training dataset: {data}")
            return data

        if use_cl and not is_eval_data:
            # augment data with negative samples for CL loss
            data = add_negative_samples(data)

        return data

    def process_data(data, with_labels: bool = True, is_eval_data: bool = False):
        # for convenience we allow `None` passed as data
        if data is None:
            return None

        # preprocess data first
        data = preprocess_data(data, is_eval_data=is_eval_data)
        # process data (create model specific instances)
        data = process_hf_dataset(
            data,
            preprocess_fn,
            stride=stride,
            max_length=max_input_length,
            remove_old_columns=True,
            tokenizer=tokenizer,
            force_preprocess=preprocess,
            keep_in_memory=keep_in_memory,
            num_processes=num_worker,
            **preprocess_fn_kwargs,
            answer_column=with_labels,
            question_column=with_labels or task in ["rc", "ap"],
            separate_answers=(
                (task in ["q", "qg", "aq", "qa", "qa2s"])
                or (task in ["ap", "rc"] and not is_eval_data)
            )
            and separate_answers,
            with_labels=with_labels,
            as_training_data=not is_eval_data,
        )
        if with_labels and task in ["q", "qg", "aq", "qa", "qa2s"]:
            # for these models we cannot handle chunks without the answer for the LM training because the question and the answer cannot be meaningfully inferred from the context
            # the negative instances can always be used since they are not used to train the LM
            if use_cl:
                # in case of cl we can use the instances without an answer in the chunk as negative instances
                data = data.map(
                    lambda x: {
                        "cl_labels": (
                            1
                            if x["cl_labels"] != 1 and not x["has_answer"]
                            else x["cl_labels"]
                        ),
                        "labels": (
                            [-100] * len(x["labels"])
                            if x["cl_labels"] != 1 and not x["has_answer"]
                            else x["labels"]
                        ),
                    },
                    load_from_cache_file=not preprocess,
                    keep_in_memory=keep_in_memory,
                    num_proc=num_worker,
                )
            # filter for instances with the answer in the chunk only which have a chance to produce a good output
            data = data.filter(
                lambda x: x["has_answer"] or ("cl_labels" in x and x["cl_labels"] == 1),
                load_from_cache_file=not preprocess,
                keep_in_memory=keep_in_memory,
                num_proc=num_worker,
            )

        if skip_retrieval or (eval_skip_retrieval and is_eval_data):
            data = data.filter(
                lambda x: x["answers"] is not None,
                load_from_cache_file=not preprocess,
                keep_in_memory=keep_in_memory,
                num_proc=num_worker,
            )
        return data

    return tokenizer, data_collator, model_init, special_token_id_map, process_data
