import logging
import random
from typing import Dict, Iterable

import numpy
import torch
from transformers import PreTrainedTokenizerFast

from ..rc.data import prepare_rc_features
from ..utils.data import get_per_sample_indices

logger = logging.getLogger(__name__)


def augment_with_negative_samples(
    samples: Dict, questions: Iterable, answers: Iterable
):
    question_pool = set(questions)
    answer_pool = set(answers)
    additional_keys = set(samples.keys()) - {"id", "question", "answers"}
    num_samples = len(samples["id"])
    # we add one column to the data to distinguish positive and negative samples
    samples["cl_labels"] = [0] * num_samples + [1] * num_samples

    for i in range(num_samples):
        # for each sample either change question, answer, or both
        samples["id"].append(f"{samples['id'][i]}_neg")
        option = random.randint(0, 2)
        question = samples["question"][i]
        answers = samples["answers"][i]
        if option == 0:
            # replace question
            question = random.choice(list(question_pool - {question}))
        elif option == 1:
            # replace answer
            answers = {
                "answer_start": [-1],
                "text": [
                    random.choice(list(answer_pool.difference(set(answers["text"]))))
                ],
            }
        elif option == 2:
            # replace question and answer
            question = random.choice(list(question_pool - {question}))
            answers = {
                "answer_start": [-1],
                "text": [
                    random.choice(list(answer_pool.difference(set(answers["text"]))))
                ],
            }
        else:
            raise ValueError(
                "Cannot have more than 3 options for creating negative samples"
            )
        samples["question"].append(question)
        samples["answers"].append(answers)
        for key in additional_keys:
            samples[key].append(samples[key][i])

    return samples


def prepare_ap_features(
    samples: Dict, *args, question_column: str = "question", **kwargs
):
    # make sure that question does not occur in examples such that ap features are built
    question_column = (
        "question"
        if isinstance(question_column, bool) and question_column
        else question_column
    )

    if question_column in samples:
        samples = {k: samples[k] for k in samples if k != question_column}

    return prepare_rc_features(
        samples, *args, question_column=question_column, **kwargs
    )


def prepare_qg_features(
    samples: Dict,
    tokenizer: PreTrainedTokenizerFast,
    seq2seq: bool,
    custom_token_type_ids: bool,
    max_length: int = None,
    stride: int = 128,
    max_question_length: int = None,
    sep_token: str = None,
    question_column: str = "question",
    context_column: str = "context",
    answer_column: str = "answers",
    fill_missing_columns: bool = True,
    verbose: bool = False,
):
    question_column = (
        "question"
        if isinstance(question_column, bool) and question_column
        else question_column
    )
    answer_column = (
        "answers"
        if isinstance(answer_column, bool) and answer_column
        else answer_column
    )
    if max_length == -1:
        max_length = None
    if max_question_length == -1:
        max_question_length = None

    if sep_token is None:
        sep_token = tokenizer.eos_token
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    if question_column in samples:
        # labelled samples including question
        sequence_1 = samples[context_column]
        if seq2seq:
            # for a sequence-to-sequence model we don't need the last separator token
            sequence_2 = [
                sep_token
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                for answer in samples[answer_column]
            ]
            targets = [
                " question:" + question + " :question"
                for question in samples[question_column]
            ]
        else:
            sequence_2 = [
                sep_token
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                + sep_token
                + " question:"
                + question
                + " :question"
                + sep_token
                for answer, question in zip(
                    samples[answer_column], samples[question_column]
                )
            ]
    else:
        sequence_1 = samples[context_column]
        if seq2seq:
            # for a sequence-to-sequence model we don't need the last separator token
            sequence_2 = [
                sep_token
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                for answer in samples[answer_column]
            ]
        else:
            sequence_2 = [
                sep_token
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                + sep_token
                for answer in samples[answer_column]
            ]
        if not seq2seq:
            if max_length is None:
                max_length = tokenizer.model_max_length
            assert max_question_length is not None
            max_length = max_length - max_question_length

    tokenized_samples = tokenizer(
        sequence_1,
        sequence_2,
        truncation="only_first",
        stride=stride,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=False,
        return_length=True,
        padding=False,
        add_special_tokens=False,
    )

    if seq2seq and question_column in samples:
        # set up tokenizer for targets in case of sequence-to-sequence models
        with tokenizer.as_target_tokenizer():
            # NOTE in case of gpt2 there is no special tokens added but seq2seq should be False anyway
            tokenized_labels = tokenizer(
                targets, padding=False, truncation=True, max_length=max_question_length
            )

    # we skip the chunks where the answer is not within the context therefore we store our positie samples in a separate dict
    processed_samples = {}
    processed_samples["input_ids"] = []
    processed_samples["attention_mask"] = []
    processed_samples["has_answer"] = []
    if custom_token_type_ids:
        processed_samples["token_type_ids"] = []

    if question_column in samples:
        processed_samples["labels"] = []
        processed_samples["question"] = []

    length = tokenized_samples.pop("length")
    overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
    (per_sample_chunk_indices,) = get_per_sample_indices(
        overflow_to_sample_mapping, range(len(overflow_to_sample_mapping))
    )

    if fill_missing_columns:
        remaining_keys = samples.keys() - processed_samples.keys()
        for _key in remaining_keys:
            processed_samples[_key] = []

    for chunk_indices in per_sample_chunk_indices:

        for i in chunk_indices:
            # sample index before tokenizing to retrieve correct information (id, context etc.)
            sample_idx = overflow_to_sample_mapping[i]

            offset_mapping = tokenized_samples["offset_mapping"][i]

            # get answer span for correct token_type_ids
            answers = samples[answer_column][sample_idx]
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0]) - 1

            # make sure that index is correct
            assert (
                samples[context_column][sample_idx][start_char : end_char + 1]
                == answers["text"][0]
            ), "Char span is wrong, make sure to run data correction first."

            # make sure that end_char is not beyond last context char (there might be index erorrs in the annotated data)
            end_char = min(end_char, len(samples["context"][sample_idx]) - 1)

            # determine whether answer is within the current chunk
            start_token = 0
            stop_token_indices = (
                torch.tensor(tokenized_samples["input_ids"][i]) == sep_token_id
            ).nonzero(as_tuple=True)[0]
            answer_index = stop_token_indices[0].item() + 1
            end_token = (
                answer_index - 2
            )  # we have to start at the last token of the context
            if (
                offset_mapping[start_token][0] <= start_char
                and offset_mapping[end_token][1] > end_char
            ):
                # answer is within current chunk
                processed_samples["has_answer"].append(True)
            else:
                # answer is not within current chunk
                processed_samples["has_answer"].append(False)

                # find start token by looping over context until char mapping is >= start_char
                while (
                    offset_mapping[start_token][1] - 1 < start_char
                ):  # additional constraint here since start char might fall into last context token
                    start_token += 1

                # find end token likewise
                while offset_mapping[end_token][0] > end_char:
                    end_token -= 1

                assert start_token <= end_token

            if fill_missing_columns:
                # add missing columns
                for _key in remaining_keys:
                    processed_samples[_key].append(samples[_key][sample_idx])

            # add sample properties to output (since we do not consider all chunks)
            processed_samples["input_ids"].append(tokenized_samples["input_ids"][i])
            processed_samples["attention_mask"].append(
                tokenized_samples["attention_mask"][i]
            )

            token_type_ids = (
                [0] * start_token
                + [1] * (end_token - start_token + 1)
                + [0] * (answer_index - end_token - 1)
            )
            if seq2seq:
                token_type_ids += [1] * (length[i] - answer_index)
            else:
                question_index = stop_token_indices[1].item() + 1
                token_type_ids += [1] * (question_index - answer_index)

            if question_column in samples:
                processed_samples["question"].append(
                    samples[question_column][sample_idx]
                )
                # add question embedding to token_type_ids
                # for the label mask everything except the question
                if seq2seq:
                    labels = tokenized_labels["input_ids"][sample_idx]
                else:
                    token_type_ids += [2] * (length[i] - question_index)
                    labels = [-100] * question_index + tokenized_samples["input_ids"][
                        i
                    ][question_index:]
                processed_samples["labels"].append(labels)
            else:
                # we add a dummy token to have a question type embedding in the end which is used in the generation process
                # thanks to the attention mask this is ignored in the model itself
                if not seq2seq:
                    token_type_ids += [2]
                    processed_samples["input_ids"][-1].append(sep_token_id)
                    processed_samples["attention_mask"][-1].append(0)

            if custom_token_type_ids:
                processed_samples["token_type_ids"].append(token_type_ids)
                assert len(processed_samples["input_ids"][-1]) == len(
                    processed_samples["token_type_ids"][-1]
                ), "Lengths of input_ids and token_type_ids must match."

    return processed_samples


def prepare_aqg_features(
    samples: Dict,
    tokenizer: PreTrainedTokenizerFast,
    with_labels: bool,
    config: str,
    seq2seq: bool,
    custom_token_type_ids: bool,
    max_gen_length: int = None,
    max_context_length: int = None,
    max_question_length: int = None,
    max_length: int = None,
    stride: int = 128,
    question_column: str = "question",
    context_column: str = "context",
    answer_column: str = "answers",
    fill_missing_columns: bool = True,
    prefix_question_whitespace: bool = True,
    as_training_data: bool = None,
):
    question_column = (
        "question"
        if isinstance(question_column, bool) and question_column
        else question_column
    )
    answer_column = (
        "answers"
        if isinstance(answer_column, bool) and answer_column
        else answer_column
    )
    # allow int values to be -1 (means None)
    if max_gen_length == -1:
        max_gen_length = None
    if max_context_length == -1:
        max_context_length = None
    if max_question_length == -1:
        max_question_length = None
    if max_length == -1:
        max_length = None

    logger.info(f"Max gen length set to {max_gen_length}")
    logger.info(f"Max context length set to {max_context_length}")
    logger.info(f"Max question length set to {max_question_length}")
    logger.info(f"Max length set to {max_length}")
    # config should be 'q', 'aq', 'qa' or 'qa2s'
    assert config in ["q", "aq", "qa", "qa2s"]
    if config == "q":
        assert answer_column in samples
    soc_token_id = tokenizer.encode("<s>", add_special_tokens=False)
    soa_token_id = tokenizer.encode("<a>", add_special_tokens=False)
    soq_token_id = tokenizer.encode("<q>", add_special_tokens=False)
    assert len(soc_token_id) == len(soa_token_id) == len(soq_token_id) == 1
    soc_token_id = soc_token_id[0]
    soa_token_id = soa_token_id[0]
    soq_token_id = soq_token_id[0]

    if max_context_length is not None:
        # Truncate contexts (from behind) to max length
        # It's ok to tokenize string and convert back since the same tokenizer (uncased or cased) is used for tokenization later on anyway
        samples[context_column] = [
            tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(context)[-max_context_length:]
            )
            for context in samples[context_column]
        ]
    if question_column in samples and max_question_length is not None:
        # Truncate questions (from behind) to max length
        # It's ok to tokenize string and convert back since the same tokenizer (uncased or cased) is used for tokenization later on anyway
        samples[question_column] = [
            tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(question)[-max_question_length:]
            )
            for question in samples[question_column]
        ]

    if question_column in samples and answer_column in samples:
        # we need questions and answers for labels
        sequence_1 = samples[context_column]
        if config == "q":
            seq_after_context_start_id = soa_token_id
            last_seq_gen_start_id = soq_token_id
            if seq2seq:
                sequence_2 = [
                    "<a>"
                    + ("" if answer["answer_start"][0] == 0 else " ")
                    + answer["text"][0]
                    for answer in samples[answer_column]
                ]
                targets = [
                    "<q>"
                    + (" " if prefix_question_whitespace else "")
                    + question
                    + "</q>"
                    for question in samples[question_column]
                ]
            else:
                sequence_2 = [
                    "<a>"
                    + ("" if answer["answer_start"][0] == 0 else " ")
                    + answer["text"][0]
                    + "<q>"
                    + (" " if prefix_question_whitespace else "")
                    + question
                    + "</q>"
                    for answer, question in zip(
                        samples[answer_column], samples[question_column]
                    )
                ]
        elif config == "aq":
            seq_after_context_start_id = soa_token_id
            last_seq_gen_start_id = soq_token_id
            _targets = [
                "<a>"
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                + "<q>"
                + (" " if prefix_question_whitespace else "")
                + question
                + "</q>"
                for answer, question in zip(
                    samples[answer_column], samples[question_column]
                )
            ]
            if seq2seq:
                targets = _targets
                sequence_2 = None
            else:
                sequence_2 = _targets
        elif config == "qa":
            seq_after_context_start_id = soq_token_id
            last_seq_gen_start_id = soa_token_id
            _targets = [
                "<q>"
                + (" " if prefix_question_whitespace else "")
                + question
                + "<a>"
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                + "</a>"
                for answer, question in zip(
                    samples[answer_column], samples[question_column]
                )
            ]
            if seq2seq:
                targets = _targets
                sequence_2 = None
            else:
                sequence_2 = _targets
        else:
            # first part of qa2s
            last_seq_gen_start_id = soq_token_id
            _targets = [
                "<q>" + (" " if prefix_question_whitespace else "") + question + "</q>"
                for question in samples[question_column]
            ]
            if seq2seq:
                targets = _targets
                sequence_2 = None
            else:
                sequence_2 = _targets

        if max_length is None:
            max_length = tokenizer.model_max_length
        max_length = (
            max_length - 1
        )  # -1 accounts for the sos token at the beginning of the sequence which is added after tokenization (to ensure every chunk starts with it)
    else:
        # inference samples
        sequence_1 = samples[context_column]
        if config == "q":
            assert answer_column in samples
            seq_after_context_start_id = soa_token_id
            last_seq_gen_start_id = soq_token_id
            if seq2seq:
                # add answer to each chunk
                sequence_2 = [
                    "<a>"
                    + ("" if answer["answer_start"][0] == 0 else " ")
                    + answer["text"][0]
                    for answer in samples[answer_column]
                ]
            else:
                # add answer + <q> to each chunk to start decoding from
                sequence_2 = [
                    "<a>"
                    + ("" if answer["answer_start"][0] == 0 else " ")
                    + answer["text"][0]
                    + "<q>"
                    for answer in samples[answer_column]
                ]
        elif config == "aq":
            seq_after_context_start_id = soa_token_id
            last_seq_gen_start_id = soa_token_id
            if seq2seq:
                sequence_2 = None
            else:
                # add <a> to each chunk to start decoding from
                sequence_2 = ["<a>" for _ in range(len(samples[context_column]))]
        elif config == "qa":
            seq_after_context_start_id = soq_token_id
            last_seq_gen_start_id = soq_token_id
            if seq2seq:
                sequence_2 = None
            else:
                # add <q> to each chunk to start decoding from
                sequence_2 = ["<q>" for _ in range(len(samples[context_column]))]
        else:
            # first part of qa2s
            # NOTE: for inference we can only prepare inputs for question generation in case of qa2s
            seq_after_context_start_id = soq_token_id
            last_seq_gen_start_id = soq_token_id
            if seq2seq:
                sequence_2 = None
            else:
                # add <q> to each chunk to start decoding from
                sequence_2 = ["<q>" for _ in range(len(samples[context_column]))]
        if max_length is None:
            max_length = tokenizer.model_max_length
        max_length = (
            max_length - 1
        )  # -1 accounts for the CLS token at the beginning of the sequence which is added later (to ensure every chunk starts with it)
        if not seq2seq:
            assert max_gen_length is not None
            max_length -= max_gen_length

    if config == "qa2s":
        max_length_2 = max_length
        if max_question_length is not None:
            # for `qa2s` we might want to set a maximum question length for the first step so that the input for the second decoding step fits into the model after generating the question (and concatenating it to the input)
            max_length -= max_question_length

    tokenized_samples = tokenizer(
        sequence_1,
        sequence_2,
        truncation="only_first",
        stride=stride,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=False,
        return_length=True,
        padding=False,
        add_special_tokens=False,
    )
    # add cls token in front of every chunk
    tokenized_samples["input_ids"] = [
        [soc_token_id] + input_ids for input_ids in tokenized_samples["input_ids"]
    ]
    tokenized_samples["attention_mask"] = [
        [1] + attention_mask for attention_mask in tokenized_samples["attention_mask"]
    ]
    tokenized_samples["length"] = [length + 1 for length in tokenized_samples["length"]]

    # compute attention mask without special tokens for sentence embeddings
    indices = [
        numpy.where(numpy.array(tokenized_samples.sequence_ids(i)) == None)[0].tolist()
        for i in range(len(tokenized_samples["input_ids"]))
    ]
    tokenized_samples["attention_mask_without_special_tokens"] = [
        torch.tensor(attention_mask)
        .index_fill(0, torch.LongTensor([0] + special_tokens_indices), 0)
        .tolist()
        for special_tokens_indices, attention_mask in zip(
            indices, tokenized_samples["attention_mask"]
        )
    ]

    if seq2seq and question_column in samples:
        # set up tokenizer for targets in case of sequence-to-sequence models
        with tokenizer.as_target_tokenizer():
            tokenized_labels = tokenizer(
                targets,
                padding=False,
                truncation=True,
                return_overflowing_tokens=False,
                add_special_tokens=False,
                max_length=max_gen_length,
            )
            # shift so that tokens < n predict n
            decoder_input_ids = [
                input_ids[:-1] for input_ids in tokenized_labels["input_ids"]
            ]
            decoder_labels = [
                input_ids[1:] for input_ids in tokenized_labels["input_ids"]
            ]

    # we skip the chunks where the answer is not within the context therefore we store our positie samples in a separate dict
    processed_samples = {}
    processed_samples["input_ids"] = []
    processed_samples["attention_mask"] = []
    processed_samples["attention_mask_without_special_tokens"] = []
    if custom_token_type_ids:
        processed_samples["token_type_ids"] = []

    if question_column in samples and answer_column in samples:
        processed_samples["labels"] = []
        processed_samples["question"] = []
        processed_samples["answers"] = []
        processed_samples["has_answer"] = []
        if seq2seq:
            # separate decoder input ids with bos token in the beginning (and missong eos token in the end)
            processed_samples["decoder_input_ids"] = []
        if "cl_labels" in samples:
            processed_samples["cl_labels"] = []
            processed_samples["cl_token_ids"] = []

    if config == "qa2s":
        # we mark samples so that we know for which step they are
        processed_samples["qa2s_step"] = []

    length = tokenized_samples.pop("length")
    overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
    (per_sample_chunk_indices,) = get_per_sample_indices(
        overflow_to_sample_mapping, range(len(overflow_to_sample_mapping))
    )

    if fill_missing_columns:
        remaining_keys = samples.keys() - processed_samples.keys()
        for _key in remaining_keys:
            processed_samples[_key] = []

    for chunk_indices in per_sample_chunk_indices:
        for i in chunk_indices:
            # sample index before tokenizing to retrieve correct information (id, context etc.)
            sample_idx = overflow_to_sample_mapping[i]

            offset_mapping = tokenized_samples["offset_mapping"][i]

            # context_end_index = tokenized_samples.sequence_ids(i).index(1) # NOTE sequence_ids is sometimes always 0?? (shouldn't happen)
            if not seq2seq or config == "q":
                input_ids_tensor = torch.tensor(tokenized_samples["input_ids"][i])
                context_end_index = (
                    input_ids_tensor == seq_after_context_start_id
                ).nonzero(as_tuple=True)[0][0].item() - 1
            else:
                context_end_index = length[i] - 1

            if question_column in samples and answer_column in samples:
                # check if answer is within current chunk so that we can mark this sample using has_answer for later filtering
                if "cl_labels" in samples and samples["cl_labels"][sample_idx] != 0:
                    # for negative samples it does not make sense to mark instances with has_answer=True
                    processed_samples["has_answer"].append(False)
                else:
                    answers = samples[answer_column][sample_idx]
                    has_answer = False
                    for start_char, text in zip(
                        answers["answer_start"], answers["text"]
                    ):
                        end_char = start_char + len(text) - 1

                        # make sure that index is correct
                        # some datasets have wrong cases hence we normalize the answers (which is done anyway in the evaluation of the model's predictions) since rectifying answers wouldn't solve the issue as it is case-sensitive
                        assert (
                            samples[context_column][sample_idx][
                                start_char : end_char + 1
                            ]
                        ) == (
                            text
                        ), f"Char span is wrong, make sure to run data correction first. Extracted answer is '{samples[context_column][sample_idx][start_char:end_char + 1]}' but given answer is '{text}'"

                        # make sure that end_char is not beyond last context char (there might be index erorrs in the annotated data)
                        end_char = min(
                            end_char, len(samples["context"][sample_idx]) - 1
                        )

                        # determine whether answer is within the current chunk
                        start_token = 0
                        # we have to start at the last token of the context since the first sep_token belongs already to the second sequence hence having offsets starting at 0 again
                        # moreover offset_mapping starts at input_ids index 1 (as well as sequence_ids)
                        end_token = context_end_index - 1
                        if (
                            offset_mapping[start_token][0] <= start_char
                            and offset_mapping[end_token][1] > end_char
                        ):
                            # answer is within current chunk
                            has_answer = True

                            # find start token by looping over context until char mapping is >= start_char
                            while (
                                offset_mapping[start_token][1] - 1 < start_char
                            ):  # additional constraint here since start char might fall into last context token
                                start_token += 1

                            # find end token likewise
                            while offset_mapping[end_token][0] > end_char:
                                end_token -= 1

                            assert start_token <= end_token

                            # we can leave the loop here since having one answer within the current chunk is enough
                            break
                    if not has_answer:
                        # no answer is within current chunk
                        if config == "q":
                            continue
                    processed_samples["has_answer"].append(has_answer)

            if fill_missing_columns:
                # add missing columns
                for _key in remaining_keys:
                    processed_samples[_key].append(samples[_key][sample_idx])

            # add sample properties to output (since we do not consider all chunks)
            processed_samples["input_ids"].append(tokenized_samples["input_ids"][i])
            processed_samples["attention_mask"].append(
                tokenized_samples["attention_mask"][i]
            )
            processed_samples["attention_mask_without_special_tokens"].append(
                tokenized_samples["attention_mask_without_special_tokens"][i]
            )

            if config == "qa2s":
                # mark instance belonging to first step
                processed_samples["qa2s_step"].append(0)

            # we always add the embedding of the next sentence in the end to start decoding with correct token_type_ids
            if not seq2seq:
                last_seq_gen_index = (
                    (input_ids_tensor == last_seq_gen_start_id)
                    .nonzero(as_tuple=True)[0][0]
                    .item()
                )
            if config == "q":
                raise NotImplementedError(
                    "Correct token type ids not implemented (not sure which answer will be considered; should probably consider all answers?)"
                )
                # account for answer within context
                token_type_ids = (
                    [0] * (start_token + 1)
                    + [1] * (end_token - start_token + 1)
                    + [0] * (context_end_index - end_token - 1)
                )
                if seq2seq:
                    token_type_ids += [1] * (length[i] - context_end_index - 1)
                else:
                    token_type_ids += [1] * (last_seq_gen_index - context_end_index - 1)
            else:
                token_type_ids = [0] * (context_end_index + 1)

            if question_column in samples and answer_column in samples:
                processed_samples["question"].append(
                    samples[question_column][sample_idx]
                )
                processed_samples["answers"].append(samples[answer_column][sample_idx])

                if seq2seq:
                    # for seq2seq models we don't have to mask any labels
                    processed_samples["decoder_input_ids"].append(
                        decoder_input_ids[sample_idx]
                    )
                    labels = decoder_labels[sample_idx]
                else:
                    # for decoder only models we have to mask the input in the labels since we don't want to learn this part
                    if config == "q":
                        # add question embedding to token_type_ids
                        token_type_ids += [2] * (length[i] - last_seq_gen_index)
                        # for the label mask everything except the question
                        # also mask the start token after the context
                        labels = [-100] * (last_seq_gen_index + 1) + tokenized_samples[
                            "input_ids"
                        ][i][last_seq_gen_index + 1 :]
                    else:
                        if config == "qa2s":
                            raise NotImplementedError()
                        # add answer+question embedding to token_type_ids
                        token_type_ids += [1] * (
                            last_seq_gen_index - context_end_index - 1
                        ) + [2] * (length[i] - last_seq_gen_index)
                        # for the label mask everything except the question & answer
                        # also mask the start token after the context
                        labels = [-100] * (context_end_index + 2) + tokenized_samples[
                            "input_ids"
                        ][i][context_end_index + 2 :]

                if "cl_labels" in samples:
                    processed_samples["cl_token_ids"].append(len(labels) - 1)
                    # add cl label
                    if config == "qa2s":
                        # for config 'qa2s' we ignore the cl_labels for the first instance (context -> q) and apply it only on the instance with answer as label
                        processed_samples["cl_labels"].append(-100)
                    else:
                        processed_samples["cl_labels"].append(
                            samples["cl_labels"][sample_idx]
                        )

                    if samples["cl_labels"][sample_idx] == 1:
                        # negative sample -> mask all labels for generation
                        labels = [-100] * len(labels)

                processed_samples["labels"].append(labels)
            else:
                # input_ids contains token to start decoding from hence we need a token_type_id for it
                if not seq2seq:
                    if config == "q":
                        token_type_ids += [2]
                    elif config == "qa2s":
                        raise NotImplementedError()
                    else:
                        token_type_ids += [1]

            if custom_token_type_ids:
                processed_samples["token_type_ids"].append(token_type_ids)

                assert len(processed_samples["input_ids"][-1]) == len(
                    processed_samples["token_type_ids"][-1]
                ), "Lengths of input_ids and token_type_ids must match."

    if config == "qa2s" and question_column in samples and answer_column in samples:
        # second step of qa2s if labels are given (otherwise input has to be prepared after question has been generated)
        sequence_1 = samples[context_column]
        if seq2seq:
            targets = [
                "<a>"
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                + "</a>"
                for answer in samples[answer_column]
            ]
            sequence_2 = [
                "<q>" + (" " if prefix_question_whitespace else "") + question
                for question in samples[question_column]
            ]
        else:
            sequence_2 = [
                "<q>"
                + (" " if prefix_question_whitespace else "")
                + question
                + "<a>"
                + ("" if answer["answer_start"][0] == 0 else " ")
                + answer["text"][0]
                + "</a>"
                for answer, question in zip(
                    samples[answer_column], samples[question_column]
                )
            ]

        # max_length is still set

        tokenized_samples = tokenizer(
            sequence_1,
            sequence_2,
            truncation="only_first",
            stride=stride,
            max_length=max_length_2,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=False,
            return_length=True,
            padding=False,
            add_special_tokens=False,
        )
        # add cls token in front of every chunk
        tokenized_samples["input_ids"] = [
            [soc_token_id] + input_ids for input_ids in tokenized_samples["input_ids"]
        ]
        tokenized_samples["attention_mask"] = [
            [1] + attention_mask
            for attention_mask in tokenized_samples["attention_mask"]
        ]
        tokenized_samples["length"] = [
            length + 1 for length in tokenized_samples["length"]
        ]

        # compute attention mask without special tokens for sentence embeddings
        indices = [
            numpy.where(numpy.array(tokenized_samples.sequence_ids(i)) == None)[
                0
            ].tolist()
            for i in range(len(tokenized_samples["input_ids"]))
        ]
        tokenized_samples["attention_mask_without_special_tokens"] = [
            torch.tensor(attention_mask)
            .index_fill(0, torch.LongTensor([0] + special_tokens_indices), 0)
            .tolist()
            for special_tokens_indices, attention_mask in zip(
                indices, tokenized_samples["attention_mask"]
            )
        ]

        if seq2seq:
            # set up tokenizer for targets in case of sequence-to-sequence models
            with tokenizer.as_target_tokenizer():
                tokenized_labels = tokenizer(
                    targets,
                    padding=False,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_gen_length,
                )
                # shift so that tokens < n predict n
                decoder_input_ids = [
                    input_ids[:-1] for input_ids in tokenized_labels["input_ids"]
                ]
            decoder_labels = [
                input_ids[1:] for input_ids in tokenized_labels["input_ids"]
            ]

        length = tokenized_samples.pop("length")
        overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
        (per_sample_chunk_indices,) = get_per_sample_indices(
            overflow_to_sample_mapping, range(len(overflow_to_sample_mapping))
        )

        for chunk_indices in per_sample_chunk_indices:
            for i in chunk_indices:
                # sample index before tokenizing to retrieve correct information (id, context etc.)
                sample_idx = overflow_to_sample_mapping[i]

                offset_mapping = tokenized_samples["offset_mapping"][i]

                # for non-seq2seq and config `qa2s` (2nd step with question in input) we need to figure out last token before question
                input_ids_tensor = torch.tensor(tokenized_samples["input_ids"][i])
                context_end_index = (input_ids_tensor == soq_token_id).nonzero(
                    as_tuple=True
                )[0][0].item() - 1

                # check if answer is within current chunk so that we can mark this sample using has_answer for later filtering
                if "cl_labels" in samples and samples["cl_labels"][sample_idx] != 0:
                    # for negative samples it does not make sense to mark instances with has_answer=True
                    processed_samples["has_answer"].append(False)
                else:
                    answers = samples[answer_column][sample_idx]
                    has_answer = False
                    for start_char, text in zip(
                        answers["answer_start"], answers["text"]
                    ):
                        end_char = start_char + len(text) - 1

                        # make sure that index is correct
                        # some datasets have wrong cases hence we normalize the answers (which is done anyway in the evaluation of the model's predictions) since rectifying answers wouldn't solve the issue as it is case-sensitive
                        assert (
                            samples[context_column][sample_idx][
                                start_char : end_char + 1
                            ]
                        ) == (
                            text
                        ), f"Char span is wrong, make sure to run data correction first. Extracted answer is '{samples[context_column][sample_idx][start_char:end_char + 1]}' but given answer is '{text}'"

                        # make sure that end_char is not beyond last context char (there might be index erorrs in the annotated data)
                        end_char = min(
                            end_char, len(samples["context"][sample_idx]) - 1
                        )

                        # determine whether answer is within the current chunk
                        start_token = 0
                        # we have to start at the last token of the context since the first sep_token belongs already to the second sequence hence having offsets starting at 0 again
                        # moreover offset_mapping starts at input_ids index 1 (as well as sequence_ids)
                        end_token = context_end_index - 1
                        if (
                            offset_mapping[start_token][0] <= start_char
                            and offset_mapping[end_token][1] > end_char
                        ):
                            # answer is within current chunk
                            has_answer = True

                            # find start token by looping over context until char mapping is >= start_char
                            while (
                                offset_mapping[start_token][1] - 1 < start_char
                            ):  # additional constraint here since start char might fall into last context token
                                start_token += 1

                            # find end token likewise
                            while offset_mapping[end_token][0] > end_char:
                                end_token -= 1

                            assert start_token <= end_token

                            # we can leave the loop here since having one answer within the current chunk is enough
                            break
                    processed_samples["has_answer"].append(has_answer)

                if fill_missing_columns:
                    # add missing columns
                    for _key in remaining_keys:
                        processed_samples[_key].append(samples[_key][sample_idx])

                # add sample properties to output (since we do not consider all chunks)
                processed_samples["input_ids"].append(tokenized_samples["input_ids"][i])
                processed_samples["attention_mask"].append(
                    tokenized_samples["attention_mask"][i]
                )
                processed_samples["attention_mask_without_special_tokens"].append(
                    tokenized_samples["attention_mask_without_special_tokens"][i]
                )

                if config == "qa2s":
                    # mark instance belonging to first step
                    processed_samples["qa2s_step"].append(1)

                # we always add the embedding of the next sentence in the end to start decoding with correct token_type_ids
                if not seq2seq:
                    last_seq_gen_index = (
                        (input_ids_tensor == last_seq_gen_start_id)
                        .nonzero(as_tuple=True)[0][0]
                        .item()
                    )
                token_type_ids = [0] * (context_end_index + 1)

                processed_samples["question"].append(
                    samples[question_column][sample_idx]
                )
                processed_samples["answers"].append(samples[answer_column][sample_idx])

                if seq2seq:
                    processed_samples["decoder_input_ids"].append(
                        decoder_input_ids[sample_idx]
                    )
                    labels = decoder_labels[sample_idx]
                else:
                    if config == "qa2s":
                        raise NotImplementedError()
                    # add answer+question embedding to token_type_ids
                    token_type_ids += [1] * (
                        last_seq_gen_index - context_end_index - 1
                    ) + [2] * (length[i] - last_seq_gen_index)
                    # for the label mask everything except the question & answer
                    # also mask the start token after the context
                    labels = [-100] * (context_end_index + 2) + tokenized_samples[
                        "input_ids"
                    ][i][context_end_index + 2 :]

                if "cl_labels" in samples:
                    processed_samples["cl_token_ids"].append(len(labels) - 1)
                    # add cl label
                    processed_samples["cl_labels"].append(
                        samples["cl_labels"][sample_idx]
                    )

                    if samples["cl_labels"][sample_idx] == 1:
                        # negative sample -> mask all labels for generation
                        labels = [-100] * len(labels)
                processed_samples["labels"].append(labels)

                if custom_token_type_ids:
                    processed_samples["token_type_ids"].append(token_type_ids)

                    assert len(processed_samples["input_ids"][-1]) == len(
                        processed_samples["token_type_ids"][-1]
                    ), "Lengths of input_ids and token_type_ids must match."
    return processed_samples
