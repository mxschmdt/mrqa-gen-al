import random
from typing import Dict, List, Mapping, Union

import nltk
import torch
from transformers import PreTrainedTokenizerFast


def mask_tokens_randomly(
    samples: Mapping, tokenizer: PreTrainedTokenizerFast, batched: bool
):
    # adapted from https://github.com/huggingface/transformers/blob/f9cde97b313c3218e1b29ea73a42414dfefadb40/examples/lm_finetuning/simple_lm_finetuning.py#L276-L301
    def random_word(
        token_ids: List[int],
        tokenizer,
        token_counts: List[int] = None,
        whole_word_masking: bool = True,
    ):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []
        if not whole_word_masking:
            # treat every token individually
            token_counts = [1] * len(token_ids)
        else:
            assert token_counts is not None

        i = 0
        for token_count in token_counts:
            token_id = token_ids[i]
            # we skip the special tokens
            if token_id in tokenizer.all_special_ids:
                output_label.append(-100)
                i += token_count
                continue

            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # append current token to output (we will predict these later)
                for token_offset in range(token_count):
                    output_label.append(token_ids[i + token_offset])

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for token_offset in range(token_count):
                        token_ids[i + token_offset] = tokenizer.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    for token_offset in range(token_count):
                        token_ids[i + token_offset] = random.choice(
                            list(tokenizer.vocab.items())
                        )[1]

                # -> rest 10% randomly keep current token
            else:
                # no masking token (will be ignored by loss function later)
                output_label.extend([-100] * token_count)
            i += token_count

        assert len(token_ids) == len(output_label)

        return token_ids, output_label

    if batched:
        all_masked_input_ids = []
        all_labels = []
        for input_ids, token_counts in zip(
            samples["input_ids"], samples["token_counts"]
        ):
            masked_input_ids, labels = random_word(input_ids, tokenizer, token_counts)
            all_masked_input_ids.append(masked_input_ids)
            all_labels.append(labels)

        return {"input_ids": all_masked_input_ids, "labels": all_labels}
    else:
        masked_input_ids, labels = random_word(
            samples["input_ids"], tokenizer, samples["token_counts"]
        )
        return {"input_ids": masked_input_ids, "labels": labels}


def prepare_mlm_features(
    samples: Mapping,
    tokenizer: PreTrainedTokenizerFast,
    as_training_data: bool,
    with_labels: bool,
    max_length: int = None,
    stride: int = 128,
    question_column: Union[str, bool, None] = "question",
    context_column: str = "context",
    answer_column: Union[str, bool, None] = "answers",
    verbose: bool = False,
):
    # pack sequences with `max_length` tokens from contiguous sentences
    # sequences start with cls token and end with sep token
    max_length = tokenizer.model_max_length if max_length == -1 else max_length
    batch_input_ids = []
    batch_tokens_per_word = []
    for context in samples[context_column]:
        input_ids = [tokenizer.cls_token_id]
        tokens_per_word = []
        sentences = nltk.sent_tokenize(context)
        for sent in sentences:
            encoding = tokenizer(
                sent,
                truncation=False,
                return_token_type_ids=False,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            sent_input_ids = encoding["input_ids"]
            words = encoding.word_ids()
            if len(input_ids) > 1 and (
                len(input_ids) + len(sent_input_ids) > max_length - 1
            ):
                # sequence reached `max_length`
                batch_input_ids.append(input_ids + [tokenizer.sep_token_id])
                input_ids = [tokenizer.cls_token_id]
                batch_tokens_per_word.append([1] + tokens_per_word + [1])
                tokens_per_word = []
            if len(sent_input_ids) > max_length - 2:
                # truncate sentence
                # NOTE: alternatively we could split the sentence
                sent_input_ids = sent_input_ids[: max_length - 2]
                words = words[: max_length - 2]
            input_ids.extend(sent_input_ids)
            tokens_per_word.extend(
                torch.unique(torch.tensor(words, dtype=torch.long), return_counts=True)[
                    1
                ].tolist()
            )
            assert (
                len(input_ids) <= max_length
            ), f"Input is longer than max tokens for model. This is probably because there exists a single sentence which is longer than max number of allowed tokens. Last sentence length (in tokens): {len(sent_input_ids)}"
        if len(input_ids) > 1:
            batch_input_ids.append(input_ids + [tokenizer.sep_token_id])
            batch_tokens_per_word.append([1] + tokens_per_word + [1])

    # return mask_tokens_randomly({'input_ids': batch_input_ids, 'token_counts': batch_tokens_per_word}, tokenizer)
    return {"input_ids": batch_input_ids, "token_counts": batch_tokens_per_word}
