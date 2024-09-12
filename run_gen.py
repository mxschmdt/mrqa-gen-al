# delay evaluation of annotation
from __future__ import annotations

import logging
from functools import partial

import comet_ml  # we need this import here so that comet_ml is loaded before PyTorch

from data.rc.model import QAModel
from data.utils.utils import monkeypatch

monkeypatch()

import math
import os

import datasets
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

from data.gen.data import prepare_ap_features, prepare_aqg_features, prepare_qg_features
from data.gen.model import get_gen_model_init
from data.rc.data import prepare_labelled_data, rectify_rc_data
from data.rc.model import get_rc_model_init
from data.utils.data import get_datasets, process_hf_dataset, unpack_samples
from data.utils.utils import (
    check_positive,
    dicts_to_feature_dict,
    expand_path,
    init_random,
    select_unique,
    setup,
    str2bool,
)
from utils.trainer import APTrainer, GenTrainer, LMFilter, RTFilter

logger = logging.getLogger(__name__)


def run(args, seed):
    # if args.num_shards is not None and args.num_shards >= 1:
    #     if args.output_dir is None:
    #         raise ValueError("You have to specify -o/--output_dir if --shards is specified")
    if args.arch in ["ap+qg", "ap+q"]:
        # answer prediction
        if args.ap_transformer is None or args.ap_model is None:
            raise ValueError(
                "You have to specify --ap_transformer and --ap_model for this task."
            )
    if args.filter == "rt":
        if args.rt_model is None or args.rt_transformer is None:
            raise ValueError(
                "You have to specify --rt_model and --rt_transformer if filter `rt` is specified"
            )

    # load data and preprocess for answer prediction
    dataset: datasets.Dataset
    dataset = get_datasets(
        args.dataset,
        args.cache_dir,
        concatenate=True,
        keep_in_memory=args.keep_in_memory,
        num_worker=args.num_worker,
    )

    # same setup as Shakeri et al.: Only contexts with >= 100 tokens, contexts truncated to 550 tokens, 100000 contexts randomly drawn

    if "answers" in dataset.column_names:
        dataset = dataset.remove_columns("answers")
    if "question" in dataset.column_names:
        dataset = dataset.remove_columns("question")
    dataset = select_unique(dataset, "context")

    if args.skip_context_length_above:
        logging.info(f"Skipping contexts > {args.skip_context_length}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.qg_model, cache_dir=args.cache_dir
        )
        try:
            dataset = dataset.filter(
                lambda x: len(
                    tokenizer.tokenize(x["context"], add_special_tokens=False)
                )
                <= args.skip_context_length,
                num_proc=args.num_worker,
            )
        except IndexError:
            logging.info("No data left after filtering for context length, exiting.")
            exit()

    if args.exclude_dataset:
        # exclude contexts for generation
        logging.info(f"Excluding contexts from specified data")
        exclude_dataset = get_datasets(
            args.exclude_dataset, args.cache_dir, concatenate=True
        )
        exclude_contexts = exclude_dataset.flatten_indices().unique("context")
        dataset = dataset.filter(
            lambda x: x["context"] not in exclude_contexts, num_proc=args.num_worker
        )

    tokenizer = AutoTokenizer.from_pretrained(args.qg_model, cache_dir=args.cache_dir)
    if args.skip_context_length_below:
        logging.info(
            f"Discarding documents with less than {args.skip_context_length_below} tokens"
        )
        dataset = dataset.filter(
            lambda x: args.skip_context_length_below
            <= len(tokenizer.tokenize(x["context"], add_special_tokens=False)),
            num_proc=args.num_worker,
        )

    num_samples = min(100000, len(dataset))
    logging.info(
        f"Randomly selecting {num_samples} documents (from {len(dataset)} available documents)"
    )
    # shuffle data using seed to make sure that we have always the same documents
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(num_samples))
    logging.info(f"Truncating documents to {550} tokens")
    # NOTE somehow this part doesn't work with multiprocessing
    dataset = dataset.map(
        lambda x: {
            "context": tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(x["context"], add_special_tokens=False)[:550]
            )
        },
        num_proc=args.num_worker,
    )

    training_args = TrainingArguments(
        output_dir="tmp",
        seed=seed,
        per_device_eval_batch_size=args.batch_size,
        disable_tqdm=False,
        use_cpu=args.use_cpu,
    )

    # generation part - always used
    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.qg_model, cache_dir=args.cache_dir
    )
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

    if args.arch in ["ap+qg", "ap+q"]:
        gen_arch = args.arch[3:]
        # answer prediction
        logging.info(f"Max answer length set to {args.max_answer_length}")

        ap = True

        gen_tokenizer._pad_token_type_id = 2  # pad with question type embedding
        if args.arch == "ap+q":
            gen_preprocess_function = partial(
                prepare_aqg_features,
                config="q",
                seq2seq=args.seq2seq,
                custom_token_type_ids=args.token_type_ids,
                max_gen_length=args.max_gen_length,
            )

            # add special tokens such that those are not split
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<a>", "<q>", "</q>"]
            }
            gen_tokenizer.add_special_tokens(special_tokens_dict)
        else:
            gen_preprocess_function = partial(
                prepare_qg_features,
                seq2seq=args.seq2seq,
                custom_token_type_ids=args.token_type_ids,
                max_question_length=args.max_gen_length,
            )

        ap_tokenizer = AutoTokenizer.from_pretrained(
            args.ap_transformer, cache_dir=args.cache_dir
        )

        ap_preprocess_function = partial(
            prepare_ap_features, max_length=args.max_input_length
        )
        ap_model = get_rc_model_init(
            path_or_name=args.ap_transformer,
            cache_dir=args.cache_dir,
            pretrained=args.ap_model,
        )()
        ap_trainer = APTrainer(
            model=ap_model,
            args=training_args,
            tokenizer=ap_tokenizer,
            num_samples=10,
            max_answer_length=args.max_answer_length,
        )
    else:
        gen_arch = args.arch
        ap = False

        gen_tokenizer._pad_token_type_id = 1  # pad with answer type embedding
        if args.arch == "aq":
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<a>", "<q>", "</q>"]
            }
        elif args.arch == "qa":
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<q>", "<a>", "</a>"]
            }
        elif args.arch == "qa2s":
            special_tokens_dict = {
                "additional_special_tokens": ["<s>", "<q>", "</q>", "<a>", "</a>"]
            }
        # add special tokens such that those are not split
        gen_tokenizer.add_special_tokens(special_tokens_dict)

        gen_preprocess_function = partial(
            prepare_aqg_features,
            config=gen_arch,
            seq2seq=args.seq2seq,
            custom_token_type_ids=args.token_type_ids,
            with_labels=False,
            # max_length=args.max_input_length,
            max_question_length=args.max_gen_length,
        )

    # data collator does not depend on task
    if args.seq2seq:
        data_collator = DataCollatorForSeq2Seq(
            gen_tokenizer, model=None, label_pad_token_id=-100
        )
    else:
        data_collator = DataCollatorForTokenClassification(
            gen_tokenizer
        )  # makes sure that labels are padded

    # we have to load the model after the tokenizer vocabulary is set (special tokens added)
    gen_model = get_gen_model_init(
        cache_dir=args.cache_dir,
        transformer=args.qg_model,
        use_cl=args.use_cl,
        tokenizer=gen_tokenizer,
    )()()
    gen_trainer = GenTrainer(
        model=gen_model,
        args=training_args,
        tokenizer=gen_tokenizer,
        config=gen_arch,
        data_collator=data_collator,
        seq2seq=args.seq2seq,
        max_gen_length=args.max_gen_length,
        custom_token_type_ids=args.token_type_ids,
    )

    # set up filter
    if args.filter == "rt":
        # rt filtering
        logging.info(f"Using RT filtering with model {args.rt_model}")
        rt_tokenizer = AutoTokenizer.from_pretrained(
            args.rt_transformer, cache_dir=args.cache_dir
        )
        rt_model = QAModel(cache_dir=args.cache_dir, model=args.rt_transformer)
        rt_model.load(
            args.rt_model,
            map_location=(
                torch.device("cpu")
                if args.nocuda or not torch.cuda.is_available()
                else None
            ),
        )
        filter = RTFilter(
            model=rt_model,
            args=training_args,
            data_collator=None,
            tokenizer=rt_tokenizer,
            max_answer_length=args.max_answer_length,
            num_worker=args.num_worker,
        )
    elif args.filter == "lm":
        # lm filtering
        logging.info("Using LM filtering with n={5}")
        filter = LMFilter(num_keep=5)
    else:
        logging.info("Filtering disabled")

    # process dataset in shards
    if args.shard_size is not None:
        # ceil so that there is a maximum of `shard_size` samples in each shard
        args.num_shards = max(1, math.ceil(len(dataset) / args.shard_size))
    if args.shards is not None:
        assert 0 <= max(args.shards) < args.num_shards
        shard_indices = args.shards
    else:
        shard_indices = range(args.num_shards)

    for i in shard_indices:
        if args.num_shards == 1:
            data = dataset
            logging.info(f"Processing {len(data)} samples.")
        else:
            data = dataset.shard(args.num_shards, i, contiguous=True)
            logging.info(
                f"Processing shard {i} ({args.num_shards} in total) with {len(data)} samples."
            )

        # move answers and question columns so that we might use them later
        data = prepare_labelled_data(data)
        if (
            "original_answers" in data.column_names
            or "original_question" in data.column_names
        ):
            data = process_hf_dataset(
                data,
                rectify_rc_data,
                answer_column="original_answers",
                question_column="original_question",
                force_preprocess=args.preprocess,
                num_processes=args.num_worker,
            )

        if ap:
            # predict answer
            data = process_hf_dataset(
                data,
                ap_preprocess_function,
                tokenizer=ap_tokenizer,
                force_preprocess=args.preprocess,
                num_processes=args.num_worker,
            )
            # predict
            predictions = ap_trainer.predict(test_dataset=data)
            data = datasets.Dataset.from_dict(dicts_to_feature_dict(predictions))

        data = process_hf_dataset(
            data,
            gen_preprocess_function,
            tokenizer=gen_tokenizer,
            force_preprocess=args.preprocess,
            num_processes=args.num_worker,
        )

        # predict
        predictions = gen_trainer.predict(test_dataset=data)

        if args.disable_filtering:
            # we use all samples without filtering
            logging.info(f"No filtering is applied, unpacking samples")
            data = unpack_samples(predictions)
        elif args.filter in ["rt", "lm"]:
            logging.info(f"Applying {args.filter} filtering")
            data = filter.filter_samples(predictions)
        else:
            logging.info(f"No filtering is applied, and samples are not unpacked")
            data = datasets.Dataset.from_dict(dicts_to_feature_dict(predictions))

        if args.output_dir is None:
            # print shard
            logging.info(f"Generated {len(data)} entries")
        elif data:
            # only store shard
            # save generated data to disk - `flatten_indices` makes sure that only data related to this shard is stored
            data.flatten_indices().save_to_disk(os.path.join(args.output_dir, str(i)))
            logging.info(f"Saved {len(data)} entries to {args.output_dir}")


if __name__ == "__main__":
    import configparser
    from argparse import ArgumentParser

    # some parameters might be in ini file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # general arguments
    parser = ArgumentParser(description="Run answer and question generation.")

    parser.add_argument(
        "--cache_dir",
        type=expand_path,
        default=config.get("Paths", "cache_dir", fallback="~/.cache"),
        help="the cache directory",
    )
    parser.add_argument(
        "--keep_in_memory",
        action="store_true",
        help="will keep preprocessed data in memory if True",
    )
    parser.add_argument(
        "--nocuda",
        action="store_true",
        help="Disables CUDA (otherwise all available GPUs will be used, so make sure to set CUDA_VISIBLE_DEVICES accordingly)",
    )
    parser.add_argument(
        "-pa",
        "--pre-allocation",
        action="store_true",
        help="Enable pre-allocation of GPU memory (this will allocate ~95%% of memory)",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "-s",
        "--seed",
        nargs="+",
        type=int,
        default=[],
        help="The seed for the random number generators",
    )
    parser.add_argument(
        "--runs",
        type=check_positive,
        default=1,
        help="The number of runs with different seeds",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="The interval for evaluation (if activated, in training steps)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Will force to preprocess any data first",
    )
    parser.add_argument(
        "--tag", nargs="+", help="Tags to be added to comet.ml experiment"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        help="The maximum input length for the model, overflowing tokens will be sliced into chunks.",
    )
    parser.add_argument(
        "--max_gen_length", type=int, help="The maximum length to generate."
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        help="The maximum length of the answer to generate in case of the `ap` architecture.",
    )
    parser.add_argument(
        "--skip_context_length_above",
        type=int,
        help="The maximum context length in tokens (contexts with more tokens will be discarded)",
    )
    parser.add_argument(
        "--skip_context_length_below",
        default=100,
        type=int,
        help="The minimum context length in tokens (contexts with less tokens will be discarded); default 100 - 0 disables it",
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        default=1,
        help="The number of worker used for preprocessing data.",
    )
    parser.add_argument(
        "--seq2seq",
        type=str2bool,
        nargs="?",
        const=True,
        required=True,
        help="Whether the model is a sequence-to-sequence model or not (labels are preprocessed differently).",
    )
    parser.add_argument(
        "--token_type_ids",
        type=str2bool,
        nargs="?",
        const=True,
        required=True,
        help="Whether the model needs token_type_ids as input and whether to create them on pre-processing.",
    )
    parser.add_argument(
        "--use_cl", action="store_true", help="Activate contrastive loss."
    )

    # logging related arguments
    parser_logging = parser.add_argument_group(title="Logging")

    parser_logging.add_argument(
        "--name", type=str, help="The file where the results will be stored"
    )
    parser_logging.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="runs",
        help="The log directory for tensorboard (e.g. for structuring experiments)",
    )

    # generation related arguments
    parser_gen = parser.add_argument_group(title="Generation")

    parser_gen.add_argument(
        "arch",
        type=str,
        help="Specifies the model architecture to use for generation",
        choices=["ap+qg", "ap+q", "aq", "qa", "qa2s"],
    )
    parser_gen.add_argument(
        "--filter",
        type=str,
        required=False,
        choices=["rt", "lm"],
        help="Specifies the method to filter generated samples",
    )
    parser_gen.add_argument(
        "--qg_model",
        type=expand_path,
        required=True,
        help="the directory where the model for question generation is stored",
    )
    parser_gen.add_argument(
        "--ap_model",
        type=expand_path,
        help="the directory where the model for answer prediction is stored",
    )
    parser_gen.add_argument(
        "--rt_model",
        type=expand_path,
        help="the directory where the model for answer prediction is stored",
    )
    parser_gen.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        metavar="dataset",
        help="the dataset(s) used for generating data",
    )
    parser_gen.add_argument(
        "--exclude_dataset",
        nargs="+",
        required=False,
        metavar="dataset",
        help="the dataset(s) of which the contexts are excluded",
    )
    parser_gen.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="the directory where the generated data will be stored",
    )
    parser_gen.add_argument(
        "--ap_transformer", help="Set the transformer model for answer prediction"
    )
    parser_gen.add_argument(
        "--rt_transformer", help="Set the transformer model for answer prediction"
    )
    parser_gen.add_argument(
        "--batch_size", type=check_positive, required=True, help="batch size (> 0)"
    )
    parser_gen.add_argument(
        "--num_shards",
        type=int,
        default=1,
        required=False,
        help="The number of shards which will be used to predict data (1 means no sharding)",
    )
    parser_gen.add_argument(
        "--shard_size", type=int, required=False, help="The size of the shards"
    )
    parser_gen.add_argument(
        "--shards",
        nargs="+",
        type=int,
        required=False,
        help="The indices of the shards to process",
    )
    parser_gen.add_argument(
        "--disable_filtering",
        action="store_true",
        required=False,
        help="Don't filter samples and unpack them",
    )
    parser_gen.add_argument(
        "--use_cpu",
        action="store_true",
        required=False,
        help="Only use CPU",
    )

    args = parser.parse_args()
    if (
        args.max_input_length is None
        and args.ap_transformer is not None
        and args.ap_transformer.startswith("SpanBERT/spanbert-")
    ):
        args.max_input_length = 512
        logging.warning(
            "Setting max_input_length for --ap-transformer to 512 as SpanBERT/spanbert-base-cased tokenizer does not have a max length."
        )

    setup(args.debug, args.pre_allocation and not args.nocuda)

    # set seed
    seed = init_random(args.seed[0] if args.seed else None)
    # call main program
    run(args, seed)
