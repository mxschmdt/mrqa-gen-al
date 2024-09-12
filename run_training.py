# delay evaluation of annotation
import copy
import dataclasses
import logging
import os
import random
from collections import defaultdict
from functools import partial

if "COMET_DISABLE_AUTO_LOGGING" not in os.environ:
    # disable cometml auto logging since it will interfere with transformers' logging
    os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

# we need this import here so that comet_ml is loaded before PyTorch
import comet_ml

from data.utils.utils import (
    PyLoggerCallback,
    get_best_std,
    monkeypatch,
    separate_map_on_prefix,
)

monkeypatch()

import torch
from datasets import disable_progress_bar
from transformers import AutoTokenizer, PreTrainedModel, Trainer

disable_progress_bar()
# import pandas as pd

from data.rc.evaluation import get_evaluator_class
from data.rc.model import get_rc_model_init
from data.utils.data import (
    d_clean_question,
    d_filter_question,
    get_datasets,
    unpack_samples,
)
from data.utils.trainer import MultiEvalTrainer
from data.utils.utils import (
    HfArgumentParser,
    check_positive,
    expand_path,
    init_random,
    setup,
)
from utils.data import (
    ActiveLearningArguments,
    ActiveLearningModelArguments,
    DataArguments,
    ModelArguments,
    TrainingArguments,
    setup_model_and_data_processing,
)
from utils.trainer import ActiveLearner, LMFilter, RTFilter


def get_rt_filter_fn(
    cache_dir,
    transformer,
    model,
    batch_size,
    seed: int = None,
    max_answer_length=None,
    truncate_question_length=None,
    num_worker=None,
):
    training_args = TrainingArguments(
        output_dir="tmp",
        seed=seed,
        per_device_eval_batch_size=batch_size,
        disable_tqdm=False,
    )
    rt_tokenizer = AutoTokenizer.from_pretrained(transformer, cache_dir=cache_dir)
    rt_model = get_rc_model_init(
        transformer=transformer, cache_dir=cache_dir, pretrained=model
    )()()
    return RTFilter(
        model=rt_model,
        args=training_args,
        data_collator=None,
        tokenizer=rt_tokenizer,
        max_answer_length=max_answer_length,
        question_max_length=truncate_question_length,
        num_worker=num_worker,
    )


def get_lm_filter_fn(num_keep: int = 5):
    return LMFilter(num_keep=num_keep)


def train(
    args,
    train_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    logging.info(f"Running training for task '{args.task}'")

    if data_args.max_input_length == -1 and model_args.transformer.startswith(
        "SpanBERT/spanbert-"
    ):
        args.max_input_length = 512
        logging.warning(
            f"Setting max_input_length to 512 as {model_args.transformer} tokenizer does not have a max length."
        )

    # add some tags for comet.ml logging
    train_args.tags.append(args.task)
    train_args.tags.append(model_args.transformer)

    if args.disable_filtering:
        filter = unpack_samples
    elif args.filter == "rt":
        # rt filtering
        if args.rt_model is None or args.rt_transformer is None:
            raise ValueError(
                "You have to specify --rt_model and --rt_transformer if filter `rt` is specified"
            )

        logging.info(f"Using RT filtering with model {args.rt_model}")
        filter = get_rt_filter_fn(
            args.cache_dir,
            args.rt_transformer,
            args.rt_model,
            train_args.per_device_eval_batch_size,
            seed=train_args.seed,
            num_worker=data_args.num_worker,
        )

    elif args.filter == "lm":
        # lm filtering
        logging.info(f"Using LM filtering with n={5}")
        filter = get_lm_filter_fn()
    else:
        filter = None

    # data
    # training data
    dataset_train = get_datasets(
        args.datasets,
        args.cache_dir,
        concatenate=True,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=filter,
        num_worker=data_args.num_worker,
    )
    if args.random_questions:
        logging.info("Randomizing questions for training dataset")
        questions = dataset_train["question"]
        dataset_train = dataset_train.map(
            lambda _: {"question": random.choice(questions)},
            num_proc=data_args.num_worker,
        )

    if args.clean_questions:
        logging.info("Cleaning questions")
        dataset_train = dataset_train.map(
            d_clean_question,
            batched=False,
            num_proc=data_args.num_worker,
            desc="Cleaning questions",
        )

    if args.filter_questions:
        logging.info("Filtering questions")
        dataset_train = dataset_train.filter(
            d_filter_question,
            batched=False,
            num_proc=data_args.num_worker,
            desc="Filtering questions",
        )

    if args.use_original_context:
        assert (
            "context_original" in dataset_train.column_names
            and "answers_context_original" in dataset_train.column_names
        )
        rename_columns_mapping = {
            "context_original": "context",
            "answers_context_original": "answers",
        }
        dataset_train = dataset_train.remove_columns(
            list(rename_columns_mapping.values())
        )
        dataset_train = dataset_train.rename_columns(rename_columns_mapping)
        logging.info(
            f"Renaming columns: {', '.join((old + ' -> ' + new) for old, new in rename_columns_mapping.items())}"
        )

    # evaluation / validation data for picking best model
    dataset_eval = get_datasets(
        args.eval,
        args.cache_dir,
        concatenate=True,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=filter,
        num_worker=data_args.num_worker,
    )
    # additional data used for evaluation during
    additional_eval_datasets = get_datasets(
        args.add_eval,
        args.cache_dir,
        concatenate=False,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=filter,
        num_worker=data_args.num_worker,
    )

    # for span prediction we should never create more instances by unpacking the answers
    keep_columns = (
        ["id", "answers", "question", "context", "title"] if args.task == "rc" else None
    )
    tokenizer, data_collator, model_init, _, process_data_fn = (
        setup_model_and_data_processing(
            dataclasses.asdict(model_args),
            args.task,
            args.cache_dir,
            data_args.stride,
            data_args.max_answer_length,
            data_args.truncate_question_length,
            data_args.max_input_length,
            data_args.num_worker,
            data_args.preprocess,
            data_args.keep_in_memory,
            data_args.unique,
            data_args.skip_question_length,
            data_args.rectify_answers,
            data_args.rectify_questions,
            True,
            False,
            args.use_cl,
            args.cl_inner_dim,
            args.skip_retrieval,
            args.eval_skip_retrieval,
            keep_columns=keep_columns,
            separate_answers=data_args.separate_answers,
        )
    )
    logging.info("Processing data for model")
    # dataset_train = dataset_train.filter(lambda sample: len(tokenizer.tokenize(sample['question'])) <= 100)
    # apply simple heuristic for filtering generated, noisy data
    # num_train_samples_raw = len(dataset_train)
    # dataset_train = dataset_train.filter(lambda sample: sample['question'].strip().endswith('?'))
    # logging.info(f"{num_train_samples_raw-len(dataset_train)} samples removed by filtering samples for which questions end with '?'")
    dataset_train = process_data_fn(dataset_train, with_labels=True, is_eval_data=False)
    # prepare evaluation data as training data in order to compute correct loss since
    # labels are sometimes wrong hence rc preprocessing will ignore labels otherwise
    # but input doesn't change anyway
    dataset_eval = process_data_fn(dataset_eval, with_labels=True, is_eval_data=False)

    if additional_eval_datasets:
        for dataset in additional_eval_datasets:
            dataset.data = process_data_fn(
                dataset.data, with_labels=True, is_eval_data=True
            )

    # set up evaluation
    if args.task in ["q", "qg", "aq", "qa", "qa2s"]:
        evaluator_class, eval_tag = None, None
        evaluator_kwargs = {}
    else:
        evaluator_class = get_evaluator_class(data_args.eval_style)
        evaluator_kwargs = dict(
            no_answer_option=not data_args.disable_no_answer,
            max_answer_length=data_args.max_answer_length,
        )
        eval_tag = data_args.eval_style + "-evaluation"
    if eval_tag is not None:
        train_args.tags.append(eval_tag)

    if args.task == "mlm":
        # we need some columns for the masking in the data collation
        train_args.remove_unused_columns = False

    # make sure that we don't save safetensors as they are not supported by model loading
    train_args.save_safetensors = False

    # logging.info(f"Training arguments: {train_args}")
    # trainer = Trainer(model=BertForQuestionAnswering.from_pretrained(model_args.transformer, cache_dir=args.cache_dir), args=training_args, train_dataset=dataset_train, eval_dataset=dataset_eval, tokenizer=tokenizer, compute_metrics=evaluator)
    logger = PyLoggerCallback()
    trainer = MultiEvalTrainer(
        add_eval_datasets=additional_eval_datasets,
        evaluator_fn=(
            partial(evaluator_class, **evaluator_kwargs)
            if evaluator_class is not None
            else None
        ),
        model_init=model_init(),
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        compute_metrics=(
            evaluator_class(dataset_eval, **evaluator_kwargs)
            if (evaluator_class is not None and dataset_eval)
            else None
        ),
        callbacks=[logger],
    )

    # evaluate (to know performance before training) and train
    print(trainer.evaluate())
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    return logger.get_logs()


def al(
    args,
    train_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ActiveLearningModelArguments,
    al_args: ActiveLearningArguments,
):
    if (
        data_args.max_input_length is None
        and model_args.rc_transformer is not None
        and model_args.rc_transformer.startswith("SpanBERT/spanbert-")
    ):
        args.max_input_length = 512
        logging.warning(
            f"Setting max_input_length to 512 as {model_args.rc_transformer} tokenizer does not have a max length."
        )

    # add tsome tags for comet.ml logging
    train_args.tags.append("al")
    if model_args.rc_transformer is not None:
        rc_train_args = copy.deepcopy(train_args)
        # set output dir
        rc_train_args.output_dir = os.path.join(rc_train_args.output_dir, "rc")
        rc_train_args.tags.append(model_args.rc_transformer)
    else:
        rc_train_args = None
    if model_args.gen_transformer is not None:
        gen_train_args = copy.deepcopy(train_args)
        # set output dir
        gen_train_args.output_dir = os.path.join(gen_train_args.output_dir, "qa2s")
        gen_train_args.tags.append(model_args.gen_transformer)
    else:
        gen_train_args = None

    logging.info(f"Running AL with tasks rc and qa2s")

    # data
    dataset_train = get_datasets(
        args.datasets,
        args.cache_dir,
        concatenate=True,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=None,
        num_worker=data_args.num_worker,
    )
    dataset_eval = get_datasets(
        args.eval,
        args.cache_dir,
        concatenate=True,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=None,
        num_worker=data_args.num_worker,
    )

    rc_model_args, gen_model_args, _ = separate_map_on_prefix(
        dataclasses.asdict(model_args), "rc_", "gen_", strict=True
    )

    if model_args.rc_transformer is not None:
        keep_columns = ["id", "answers", "question", "context", "title"]
        rc_tokenizer, rc_data_collator, rc_model_init, _, rc_process_data_fn = (
            setup_model_and_data_processing(
                rc_model_args,
                "rc",
                args.cache_dir,
                data_args.stride,
                data_args.max_answer_length,
                data_args.truncate_question_length,
                data_args.max_input_length,
                data_args.num_worker,
                data_args.preprocess,
                data_args.keep_in_memory,
                data_args.unique,
                data_args.skip_question_length,
                data_args.rectify_answers,
                data_args.rectify_questions,
                True,
                False,
                args.use_cl,
                args.cl_inner_dim,
                args.skip_retrieval,
                args.eval_skip_retrieval,
                separate_answers=data_args.separate_answers,
            )
        )
    else:
        rc_tokenizer, rc_data_collator, rc_model_init, rc_process_data_fn = (
            None,
            None,
            None,
            None,
        )
    if model_args.gen_transformer is not None:
        (
            gen_tokenizer,
            gen_data_collator,
            gen_model_init,
            gen_special_token_id_map,
            gen_process_data_fn,
        ) = setup_model_and_data_processing(
            gen_model_args,
            "qa2s",
            args.cache_dir,
            data_args.stride,
            data_args.max_answer_length,
            data_args.truncate_question_length,
            data_args.max_input_length,
            data_args.num_worker,
            data_args.preprocess,
            data_args.keep_in_memory,
            data_args.unique,
            data_args.skip_question_length,
            data_args.rectify_answers,
            data_args.rectify_questions,
            True,
            False,
            args.use_cl,
            args.cl_inner_dim,
            args.skip_retrieval,
            args.eval_skip_retrieval,
            separate_answers=data_args.separate_answers,
        )
    else:
        (
            gen_tokenizer,
            gen_data_collator,
            gen_model_init,
            gen_special_token_id_map,
            gen_process_data_fn,
        ) = (None, None, None, None, None)

    logging.info("Processing eval data")
    # rc evaluation expects an extracted answer and it doesnt matter whether we prepare labels or not
    if rc_process_data_fn is not None:
        rc_dataset_eval = rc_process_data_fn(
            dataset_eval, with_labels=True, is_eval_data=False
        )
    else:
        rc_dataset_eval = None
    if gen_process_data_fn is not None:
        gen_dataset_eval = gen_process_data_fn(
            dataset_eval, with_labels=True, is_eval_data=True
        )
    else:
        gen_dataset_eval = None

    # set up evaluation
    rc_evaluator = get_evaluator_class(data_args.eval_style)(
        rc_dataset_eval,
        no_answer_option=not data_args.disable_no_answer,
        max_answer_length=data_args.max_answer_length,
    )
    rc_train_args.tags.append(data_args.eval_style)

    rc_logger, gen_logger = PyLoggerCallback(), PyLoggerCallback()
    # requires model_init because we start with given model in each iteration of AL (cold start)
    # calling model_init will make sure to load checkpoint if wanted
    learner = ActiveLearner(
        gen_model_init=gen_model_init() if gen_model_init is not None else None,
        gen_tokenizer=gen_tokenizer,
        gen_data_collator=gen_data_collator,
        gen_eval_dataset=gen_dataset_eval,
        gen_process_data_fn=gen_process_data_fn,
        gen_compute_metrics=None,
        gen_args=gen_train_args,
        rc_model_init=rc_model_init() if rc_model_init is not None else None,
        rc_tokenizer=rc_tokenizer,
        rc_data_collator=rc_data_collator,
        rc_eval_dataset=rc_dataset_eval,
        rc_process_data_fn=rc_process_data_fn,
        rc_compute_metrics=rc_evaluator,
        rc_args=rc_train_args,
        max_gen_length=data_args.truncate_question_length,
        gen_special_token_id_map=gen_special_token_id_map,
        output_dir=train_args.output_dir,
        rc_callbacks=[rc_logger],
        gen_callbacks=[gen_logger],
    )
    learner.run_al(
        data=dataset_train,
        rounds=al_args.rounds,
        num_samples=al_args.samples,
        mode=al_args.mode,
    )
    return rc_logger.get_logs()


def eval(
    args,
    train_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    logging.info(f"Running evaluation for task '{args.task}'")

    if data_args.max_input_length == -1 and model_args.transformer.startswith(
        "SpanBERT/spanbert-"
    ):
        args.max_input_length = 512
        logging.warning(
            f"Setting max_input_length to 512 as {model_args.transformer} tokenizer does not have a max length."
        )

    # add some tags for comet.ml logging
    train_args.tags.append(args.task)
    train_args.tags.append(model_args.transformer)

    # data
    # evaluation / validation data for picking best model
    dataset_eval = get_datasets(
        args.datasets,
        args.cache_dir,
        concatenate=True,
        keep_in_memory=data_args.keep_in_memory,
        num_worker=data_args.num_worker,
    )

    # for span prediction we should never create more instances by unpacking the answers
    keep_columns = (
        ["id", "answers", "question", "context", "title"] if args.task == "rc" else None
    )
    tokenizer, data_collator, model_init, _, process_data_fn = (
        setup_model_and_data_processing(
            dataclasses.asdict(model_args),
            args.task,
            args.cache_dir,
            data_args.stride,
            data_args.max_answer_length,
            data_args.truncate_question_length,
            data_args.max_input_length,
            data_args.num_worker,
            data_args.preprocess,
            data_args.keep_in_memory,
            data_args.unique,
            data_args.skip_question_length,
            data_args.rectify_answers,
            data_args.rectify_questions,
            True,
            False,
            args.use_cl,
            args.cl_inner_dim,
            args.skip_retrieval,
            False,
            keep_columns=keep_columns,
            separate_answers=data_args.separate_answers,
        )
    )
    logging.info("Processing data for model")
    # prepare evaluation data as training data in order to compute correct loss since
    # labels are sometimes wrong hence rc preprocessing will ignore labels otherwise
    # but input doesn't change anyway
    dataset_eval = process_data_fn(dataset_eval, with_labels=True, is_eval_data=True)

    # set up evaluation
    if args.task in ["q", "qg", "aq", "qa", "qa2s"]:
        evaluator_class, eval_tag = None, None
    else:
        evaluator_class = get_evaluator_class(data_args.eval_style)
        evaluator_kwargs = dict(
            no_answer_option=not data_args.disable_no_answer,
            max_answer_length=data_args.max_answer_length,
        )
        evaluator_kwargs.update(
            prediction_file=os.path.join(train_args.output_dir, "pred.json")
        )
        eval_tag = data_args.eval_style + "-evaluation"
    if eval_tag is not None:
        train_args.tags.append(eval_tag)

    if args.task == "mlm":
        # we need some columns for the masking in the data collation
        train_args.remove_unused_columns = False

    model = model_init()()
    if model_args.pretrained is not None and not isinstance(model, PreTrainedModel):
        model.load(model_args.pretrained)
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=(
            evaluator_class(dataset_eval, **evaluator_kwargs)
            if (evaluator_class is not None and dataset_eval)
            else None
        ),
    )
    # evaluate (to know performance before training) and train
    metrics = trainer.evaluate(eval_dataset=dataset_eval)
    logging.info(metrics)
    return metrics


if __name__ == "__main__":

    def main():
        import argparse
        import configparser

        # some parameters might be in ini file
        config = configparser.ConfigParser()
        config.read("config.ini")

        ### parent parsers

        # general parser
        parser_general = argparse.ArgumentParser(
            description="A parser including general arguments", add_help=False
        )  # can be used as parent for other parsers
        parser_general.add_argument(
            "--cache_dir",
            type=expand_path,
            default=config.get("Paths", "cache_dir", fallback="~/.cache"),
            help="the cache directory",
        )
        parser_general.add_argument(
            "-pa",
            "--pre-allocation",
            action="store_true",
            help="Enable pre-allocation of GPU memory (this will allocate 95%% of memory)",
        )
        parser_general.add_argument(
            "--runs",
            type=check_positive,
            default=1,
            help="The number of runs (usually with different seeds)",
        )
        # cl related arguments
        parser_general.add_argument(
            "--cl_inner_dim",
            type=int,
            default=128,
            help="The inner dim of the CL head (between hidden state and binary output)",
        )
        parser_general.add_argument(
            "--use_cl", action="store_true", help="Activate contrastive loss."
        )
        # filtering related arguments
        parser_general.add_argument(
            "--filter",
            type=str,
            required=False,
            choices=["rt", "lm"],
            help="Specifies the model architecture to use for generation",
        )
        parser_general.add_argument(
            "--rt-model",
            type=expand_path,
            help="the directory where the model for answer prediction is stored",
        )
        parser_general.add_argument(
            "--rt-transformer", help="Set the transformer model for answer prediction"
        )
        parser_general.add_argument(
            "--disable-filtering",
            action="store_true",
            required=False,
            help="Don't filter samples and unpack them",
        )

        # training related arguments
        parser_training = argparse.ArgumentParser(
            description="A parser used for training a model.",
            add_help=False,
            parents=[parser_general],
        )  # can be used as parent for other parsers
        parser_training.add_argument(
            "--skip_retrieval",
            action="store_true",
            help="Use instance with answer only if set to True, otherwise all documents (affects training data only)",
        )
        parser_training.add_argument(
            "--eval_skip_retrieval",
            action="store_true",
            help="Use instance with answer only if set to True, otherwise all documents (affects evaluation data only)",
        )
        parser_training.add_argument(
            "--single_document",
            action="store_true",
            help="Keep single document in case only answerable documents are used (affects training data only)",
        )
        parser_training.add_argument(
            "--eval_single_document",
            action="store_true",
            help="Keep single document in case only answerable documents are used (affects evaluation data only)",
        )
        parser_training.add_argument(
            "--datasets",
            required=True,
            nargs="+",
            metavar="dataset",
            help="the dataset(s) used for training the model",
        )
        parser_training.add_argument(
            "--eval",
            "--eval-datasets",
            nargs="+",
            metavar="dataset",
            help="the dataset(s) used for evaluating the model and selecting the best model checkpoint",
        )
        parser_training.add_argument(
            "--add-eval",
            "--add-eval-datasets",
            nargs="+",
            metavar="dataset",
            help="the additional dataset(s) used for evaluating the model",
        )
        parser_training.add_argument(
            "--random-questions",
            action="store_true",
            help="Randomize questions for training data",
        )
        parser_training.add_argument(
            "--clean-questions",
            action="store_true",
            help="Clean questions (using handcrafted rules) for training data",
        )
        parser_training.add_argument(
            "--filter-questions",
            action="store_true",
            help="Filter questions (using handcrafted rules) for training data",
        )
        parser_training.add_argument(
            "--use-original-context",
            action="store_true",
            help="Use original context if available",
        )

        # main parser
        parser = HfArgumentParser(
            description="Run RC training and prediction as well as AP, Q, QG, AQ, QA & QA2S training.",
            allow_abbrev=False,
        )
        subparsers = parser.add_subparsers(title="command", dest="cmd", required=True)

        ### parser commands

        # train command
        parser_train = subparsers.add_parser(
            "train",
            dataclass_types=[TrainingArguments, DataArguments, ModelArguments],
            description="Runs training on the given datasets.",
            parents=[parser_training],
        )
        parser_train.add_argument(
            "task",
            type=str,
            help="Specifies the task to perform",
            choices=["mlm", "rc", "ap", "q", "qg", "aq", "qa", "qa2s"],
        )
        parser_train.set_defaults(func=train)

        # train command
        parser_al = subparsers.add_parser(
            "al",
            dataclass_types=[
                TrainingArguments,
                DataArguments,
                ActiveLearningModelArguments,
                ActiveLearningArguments,
            ],
            description="Runs active learning.",
            parents=[parser_training],
        )
        parser_al.set_defaults(func=al)

        # eval command
        parser_eval = subparsers.add_parser(
            "eval",
            dataclass_types=[TrainingArguments, DataArguments, ModelArguments],
            description="Runs evaluation for the given model on the given datasets.",
            parents=[parser_general],
        )
        parser_eval.set_defaults(func=eval)

        parser_eval.add_argument(
            "task", type=str, help="Specifies the task to perform", choices=["rc", "ap"]
        )
        # parser_eval.add_argument('model', type=expand_path, help='the directory where the model and other files are stored (for training & evaluation)')
        parser_eval.add_argument(
            "--datasets",
            nargs="+",
            metavar="dataset",
            required=True,
            help="the dataset(s) used for evaluating the model",
        )
        parser_eval.add_argument(
            "--skip-retrieval",
            action="store_true",
            help="Use instance with answer only if set to True, otherwise all documents",
        )
        parser_eval.add_argument(
            "--single-document",
            action="store_true",
            help="Keep single document in case only answerable documents are used",
        )
        parser_eval.add_argument(
            "--eval-seed",
            type=int,
            default=1234,
            help="the seed for uniformly drawing samples for evaluation",
        )

        all_args = parser.parse_args_into_dataclasses()
        train_args = all_args[0]
        data_args = all_args[1]
        args = all_args[-1]
        other_args = all_args[2:-1]
        # do setup before any logging to make sure that no default handler is created
        setup(train_args.debug, args.pre_allocation and not train_args.no_cuda)

        # do multiple runs
        seeds_from_arg = train_args.seed
        seeds = []
        metrics_all = defaultdict(list)
        for run in range(args.runs):
            logging.info("===== Run %d/%d =====", run + 1, args.runs)
            # set seed
            seed = init_random(
                seeds_from_arg[run]
                if seeds_from_arg is not None and len(seeds_from_arg) > run
                else None
            )
            seeds.append(seed)

            # call command-specific function
            train_args.logging_dir = os.path.join(
                train_args.output_dir, "runs", str(seed)
            )
            train_args.seed = seed
            metrics = args.func(args, train_args, data_args, *other_args)
            # aggregate metrics from multiple runs
            for key, value in metrics.items():
                metrics_all[key].append(value)

        logging.info(
            f"Metrics from {args.runs} run{'s' if args.runs > 1 else ''}: {get_best_std(metrics_all, None)}"
        )
        # extract best model checkpoint for each run according to `metric_for_best_model` and return mean and std dev over all evaluation datasets
        if train_args.metric_for_best_model is not None:
            # extract best mdoel checkpoint for each run
            print(train_args.metric_for_best_model)
            print(metrics_all[f"eval/{train_args.metric_for_best_model}"])
            best_model_idx = torch.tensor(
                metrics_all[f"eval/{train_args.metric_for_best_model}"]
            ).argmax(dim=1)
            logging.info(
                f"Evaluation metrics (using best model checkpoint w.r.t. eval/{train_args.metric_for_best_model}) from {args.runs} run{'s' if args.runs > 1 else ''}: {get_best_std(metrics_all, best_model_idx)}"
            )

    main()
