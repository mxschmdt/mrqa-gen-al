import logging
from typing import Dict, Mapping, Union

import numpy
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from ..utils.data import find_answer_span, get_per_sample_indices


def rectify_rc_data(
    samples: Dict,
    context_column: str = "context",
    question_column: str = "question",
    answer_column: str = "answers",
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
    if (
        answer_column is not None
        and answer_column in samples
        and "answer_start" not in samples[answer_column][0]
    ):
        # cannot rectify answers
        answer_column = None

    # iterate over samples and fix answer span and allow to skip sample if answer text is not found within context
    keys = set(samples.keys())
    rectified_samples = {key: [] for key in keys}
    if answer_column is not None and answer_column in samples:
        keys.remove(answer_column)
    if question_column is not None and question_column in samples:
        keys.remove(question_column)

    for i in range(len(samples[next(iter(samples))])):
        context = samples[context_column][i]

        answers_new = {"answer_start": [], "text": []}
        if answer_column is not None and answer_column in samples:
            # rectify answers
            answers = samples[answer_column][i]

            for start_char, text in zip(answers["answer_start"], answers["text"]):
                end_char = start_char + len(text) - 1

                if text != context[start_char : end_char + 1]:
                    if text.lower() == context[start_char : end_char + 1].lower():
                        # sometimes the answer is just wrongly cased
                        text = context[start_char : end_char + 1]
                    else:
                        corrected_start_char, corrected_end_char = find_answer_span(
                            context, text, start_char
                        )
                        if verbose:
                            if corrected_start_char == corrected_end_char == -1:
                                logging.warning(
                                    "Answer not found in context, setting span start to -1."
                                )
                            else:
                                logging.warning(
                                    f"Sample {samples['id'][i]}: Corrected span for answer '{text}': ({start_char, end_char} -> {corrected_start_char, corrected_end_char})"
                                )
                        start_char = corrected_start_char
                # add (potentially corrected) answer
                answers_new["answer_start"].append(start_char)
                answers_new["text"].append(text)

        if (
            answers_new["answer_start"]
            or answer_column is None
            or answer_column not in samples
        ):
            # keep sample
            for key in keys:
                rectified_samples[key].append(samples[key][i])

            if answer_column is not None and answer_column in samples:
                rectified_samples[answer_column].append(answers_new)

            if question_column is not None and question_column in samples:
                # remove surrounding whitespace from question
                rectified_samples[question_column].append(
                    samples[question_column][i].strip()
                )
    return rectified_samples


def prepare_labelled_data(
    data: Dataset,
    answer_column: str = "answers",
    question_column: str = "question",
    answer_column_move_to: str = "original_answers",
    question_colum_move_to: str = "original_question",
):
    def move_answers_questions(sample: Dict):
        if answer_column is not None and answer_column in sample:
            sample[answer_column_move_to] = sample[answer_column]
            del sample[answer_column]
        if question_column is not None and question_column in sample:
            sample[question_colum_move_to] = sample[question_column]
            del sample[question_column]
        return sample

    return data.map(move_answers_questions)


def prepare_rc_features(
    samples: Mapping,
    tokenizer: PreTrainedTokenizerFast,
    as_training_data: bool,
    with_labels: bool,
    question_max_length: int = None,
    max_length: int = None,
    stride: int = 128,
    question_column: Union[str, bool, None] = "question",
    context_column: str = "context",
    answer_column: Union[str, bool, None] = "answers",
    fill_missing_columns: bool = True,
    verbose: bool = False,
):
    # NOTE stride has to be smaller than maximum length allocated for context (i.e. max length - question - 3 (in case of Bert special tokens))
    # `with_labels` never changes the input for RC hence we can always add labels no matter if `as_training_data`
    question_column = (
        "question"
        if question_column is None
        or isinstance(question_column, bool)
        and question_column
        else question_column
    )
    answer_column = (
        "answers"
        if answer_column is None or isinstance(answer_column, bool) and answer_column
        else answer_column
    )

    if question_column in samples:
        # rc sample
        if question_max_length is not None and question_max_length != -1:
            # truncate questions to max length
            sentence_1 = []
            for question in samples[question_column]:
                # It's ok to tokenize string and convert back since the same tokenizer (uncased or cased) is used for tokenization later on anyway
                sentence_1.append(
                    tokenizer.convert_tokens_to_string(
                        tokenizer.tokenize(question)[-question_max_length:]
                    )
                )
        else:
            sentence_1 = samples[question_column]
        sentence_2 = samples[context_column]

        truncation = "only_second"  # don't truncate question
        context_embedding_id = 1
    else:
        # ap sample
        sentence_1 = samples[context_column]
        sentence_2 = None
        truncation = True  # there is only one input sequence
        context_embedding_id = 0

    # additional keys
    add_question = question_column in samples

    # tokenize pair of sequences
    tokenized_samples = tokenizer(
        sentence_1,
        sentence_2,
        truncation=truncation,
        max_length=None if max_length == -1 else max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_length=True,
        padding=False,
    )

    # compute attention mask without special tokens for sentence embeddings
    indices = [
        numpy.where(numpy.array(tokenized_samples.sequence_ids(i)) == None)[0].tolist()
        for i in range(len(tokenized_samples["input_ids"]))
    ]
    tokenized_samples["attention_mask_without_special_tokens"] = [
        torch.tensor(attention_mask)
        .index_fill(0, torch.LongTensor(special_tokens_indices), 0)
        .tolist()
        for special_tokens_indices, attention_mask in zip(
            indices, tokenized_samples["attention_mask"]
        )
    ]

    tokenized_samples["id"] = []
    tokenized_samples["context"] = []
    tokenized_samples["context_offset"] = []

    # conditional keys
    if add_question:
        tokenized_samples["question"] = []

    # special columns for training samples
    if with_labels:
        assert (
            answer_column in samples
        ), "Processing samples with labels but label not found"
        tokenized_samples["label_ids"] = []
    use_char_spans = with_labels and "answer_start" in samples["answers"][0]

    if as_training_data:
        # for training data we need labels
        assert use_char_spans

    # special columns for samples with answers
    if with_labels:
        tokenized_samples["extracted_answer"] = []
        tokenized_samples["answers"] = []
    if use_char_spans:
        tokenized_samples["char_spans"] = []

    overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
    (per_sample_chunk_indices,) = get_per_sample_indices(
        overflow_to_sample_mapping, range(len(overflow_to_sample_mapping))
    )

    if fill_missing_columns:
        remaining_keys = samples.keys() - tokenized_samples.keys()
        for _key in remaining_keys:
            tokenized_samples[_key] = []

    for chunk_indices in per_sample_chunk_indices:
        # same extracted answer for all chunks per sample (has to be a pointer)
        extracted_answer = []
        # variable used to enable check if answers can potentially be extracted (disabled if spans are off)
        can_extract_answers = True

        for i in chunk_indices:
            # sample index before tokenizing to retrieve correct information (id, context etc.)
            sample_idx = overflow_to_sample_mapping[i]

            # set context offset so that we know which tokens we have to consider for answer extraction
            context_offset = tokenized_samples.sequence_ids(i).index(
                context_embedding_id
            )
            # check that we have valid inputs for our task
            if question_column in samples and not 1 in tokenized_samples.sequence_ids(
                i
            ):
                raise ValueError(
                    "The tokenizer does not provide valid token type ids. Please make sure to use a tokenizer which can handle Question Answering inputs."
                )

            # compute labels if answer is given
            if with_labels:
                offset_mapping = tokenized_samples["offset_mapping"][i]
                answers = samples[answer_column][sample_idx]

                if len(answers["text"]) == 0:
                    # no answer for this sample -> set label to cls token
                    tokenized_samples["label_ids"].append([0, 0])
                    tokenized_samples["answers"].append(None)
                    tokenized_samples["extracted_answer"].append(None)
                    if use_char_spans:
                        tokenized_samples["char_spans"].append(None)
                else:
                    if as_training_data:
                        # compute start and end token indices
                        start_char = answers["answer_start"][0]
                        end_char = start_char + len(answers["text"][0]) - 1

                        # make sure that index is correct
                        assert (
                            samples[context_column][sample_idx][
                                start_char : end_char + 1
                            ]
                            == answers["text"][0]
                        ), f"Char span is wrong, make sure to run data correction first. (label is '{answers['text'][0]}', char span is {(start_char,end_char)}, extracted answer is '{samples[context_column][sample_idx][start_char:end_char + 1]}')"

                        # make sure that end_char is not beyond last context char (there might be index erorrs in the annotated data)
                        end_char = min(
                            end_char, len(samples["context"][sample_idx]) - 1
                        )

                        # determine whether answer is within the current chunk
                        start_token = context_offset
                        end_token = len(offset_mapping) - 2

                        if not (
                            offset_mapping[start_token][0] <= start_char
                            and offset_mapping[end_token][1] > end_char
                        ):
                            # answer is not within current chunk -> set label to cls token
                            tokenized_samples["label_ids"].append([0, 0])
                        else:
                            # find start token by looping over context until char mapping is >= start_char
                            while (
                                offset_mapping[start_token][1] - 1 < start_char
                            ):  # additional constraint here since start char might fall into last context token
                                start_token += 1

                            # find end token likewise
                            while offset_mapping[end_token][0] > end_char:
                                end_token -= 1

                            assert start_token <= end_token

                            # sometimes data is annotated such that start/end char is within token; in this case we set the start/end char to the first/last char of the token into which the start/end char falls
                            # this will not influnce the training labels since the model predicts spans in terms of tokens but may decrease the evaluation score (but usually normalization takes care of it)
                            start_char_from_mapping, end_char_from_mapping = (
                                offset_mapping[start_token][0],
                                offset_mapping[end_token][1] - 1,
                            )
                            if (start_char, end_char) != (
                                start_char_from_mapping,
                                end_char_from_mapping,
                            ):
                                if verbose:
                                    logging.warning(
                                        f"Sample {samples['id'][sample_idx]} with gold answer '{answers['text'][0]}': char span start/end is within token, correcting span ({start_char, end_char} -> {start_char_from_mapping, end_char_from_mapping}; '{samples[context_column][sample_idx][start_char:end_char+1]}' -> '{samples[context_column][sample_idx][start_char_from_mapping:end_char_from_mapping+1]}')"
                                    )
                                start_char = start_char_from_mapping
                                end_char = end_char_from_mapping

                            if (
                                samples[context_column][sample_idx][
                                    start_char : end_char + 1
                                ]
                                not in extracted_answer
                            ):
                                extracted_answer.append(
                                    samples[context_column][sample_idx][
                                        start_char : end_char + 1
                                    ]
                                )

                            tokenized_samples["label_ids"].append(
                                [start_token, end_token]
                            )

                        # add answer span
                        tokenized_samples["char_spans"].append(
                            [
                                (answer_start, answer_start + len(answer) - 1)
                                for answer, answer_start in zip(
                                    answers["text"], answers["answer_start"]
                                )
                            ]
                        )
                    else:
                        # answer span not given
                        if use_char_spans:
                            tokenized_samples["char_spans"].append(
                                [None] * len(answers["text"])
                            )
                        can_extract_answers = False
                        tokenized_samples["label_ids"].append([-100, -100])

                    # add answer text
                    tokenized_samples["answers"].append(answers["text"])

                    # every chunk will have an extracted answer (the same for the sample)
                    tokenized_samples["extracted_answer"].append(extracted_answer)

            # add to tokenized_samples dict after potentially skipping samples
            if fill_missing_columns:
                # add missing columns
                for _key in remaining_keys:
                    tokenized_samples[_key].append(samples[_key][sample_idx])

            tokenized_samples["context_offset"].append(context_offset)

            # we need the id for evaluation
            tokenized_samples["id"].append(samples["id"][sample_idx])

            # context is used for extracting the answer
            tokenized_samples["context"].append(samples["context"][sample_idx])

            # add conditional information
            if add_question:
                tokenized_samples["question"].append(samples["question"][sample_idx])

        if answer_column in samples and can_extract_answers:
            # answer has to be extracted in at least one chunk
            if not extracted_answer:
                logging.warning(
                    f"No answer for sample {samples['id'][sample_idx]} could be extracted. This is probably due to the answer not being part of any chunk."
                )

    return tokenized_samples
