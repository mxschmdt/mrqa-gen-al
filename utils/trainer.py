import copy
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import takewhile
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from evaluate import load as load_metric
from tqdm import tqdm
from transformers import Trainer

from data.rc.data import prepare_rc_features
from data.rc.evaluation import RCMetric, rc_predict
from data.utils.data import find_answer_span, unpack_samples
from data.utils.utils import PyLoggerCallback, separate_map_on_prefix

logger = logging.getLogger(__name__)


class RCTrainer(Trainer):
    def __init__(
        self,
        max_answer_length: int,
        num_worker: int = 1,
        question_max_length: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_worker = num_worker
        self.max_answer_length = max_answer_length
        self.question_max_length = question_max_length

    def preprocess_and_predict_sample(self, sample: Dataset):
        # do not use the answer column for preparing the features in order to do inference only
        feats = sample.map(
            prepare_rc_features,
            batched=True,
            keep_in_memory=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "question_column": "question",
                "answer_column": None,
                "question_max_length": self.question_max_length,
                "as_training_data": False,
                "with_labels": False,
            },
            num_proc=self.num_worker,
        )
        # apply fix for transformers trainer removing first dimension if batch-size is 1
        feats = feats.map(
            lambda x: {"transformers_fix": True},
            keep_in_memory=True,
            num_proc=self.num_worker,
        )
        # do prediction
        return self.predict_sample(feats)

    def predict_sample(self, sample: Dataset):
        logits = self.predict(sample).predictions
        if logits.ndim == 2:
            # add batch dimension
            logits = np.expand_dims(logits, 0)
        preds = rc_predict(
            logits,
            sample["id"],
            sample["id"],
            sample["context"],
            sample["context_offset"],
            sample["length"],
            sample["offset_mapping"],
            1,
            max_length=self.max_answer_length,
        )
        return list(preds.values())[0].extract_answers(1)[1][0].strip()


class APTrainer(Trainer):
    def __init__(self, num_samples, max_answer_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.max_answer_length = max_answer_length

    def predict(self, test_dataset: Dataset):
        logits = super().predict(test_dataset).predictions

        additional_info = defaultdict(dict)
        for sample in test_dataset:
            if "original_answers" in sample:
                additional_info[sample["id"]]["original_answers"] = sample[
                    "original_answers"
                ]
            if "original_question" in sample:
                additional_info[sample["id"]]["original_question"] = sample[
                    "original_question"
                ]

        preds = rc_predict(
            logits,
            test_dataset["id"],
            test_dataset["id"],
            test_dataset["context"],
            test_dataset["context_offset"],
            test_dataset["length"],
            test_dataset["offset_mapping"],
            self.num_samples,
            max_length=self.max_answer_length,
        )
        predictions = []
        for _instance_id, pred in preds.items():

            # context = self.tokenizer.decode(input_ids[sample_id][input_ids[sample_id] != -100], skip_special_tokens=True)
            context = pred.contexts[_instance_id]
            _, answers, spans = pred.extract_answers(self.num_samples)

            # add all predicted answers
            for answer, span in zip(answers, spans):
                predictions.append(
                    dict(
                        **additional_info[_instance_id],
                        **{
                            "id": _instance_id,
                            "context": context,
                            "answers": {
                                "text": [answer],
                                "answer_start": [span[0]],
                            },
                        },
                    )
                )
        return predictions


class GenTrainer(Trainer):
    def __init__(
        self,
        config,
        seq2seq: bool,
        max_gen_length: int,
        custom_token_type_ids: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # config should be in ['qg', 'q', 'aq', 'qa', 'qa2s']
        assert config in [
            "qg",
            "q",
            "aq",
            "qa",
            "qa2s",
        ], f"Config is {config} but should be on of {['qg', 'q', 'aq', 'qa', 'qa2s']}"
        if config == "qa2s" and max_gen_length is None:
            raise ValueError("`max_gen_length` cannot be None for 'qa2s'")
        self.config = config
        self.seq2seq = seq2seq
        self.max_gen_length = max_gen_length
        self.custom_token_type_ids = custom_token_type_ids

        # set some token ids
        if self.config == "q":
            soq_token_id = self.tokenizer.encode("<q>", add_special_tokens=False)
            eoq_token_id = self.tokenizer.encode("</q>", add_special_tokens=False)
            assert len(eoq_token_id) == len(soq_token_id) == 1
            self.soq_token_id = soq_token_id[0]
            self.eoq_token_id = eoq_token_id[0]
        elif self.config in ["aq", "qa", "qa2s"]:
            soq_token_id = self.tokenizer.encode("<q>", add_special_tokens=False)
            soa_token_id = self.tokenizer.encode("<a>", add_special_tokens=False)
            assert len(soq_token_id) == len(soa_token_id) == 1
            self.soq_token_id, self.soa_token_id = soq_token_id[0], soa_token_id[0]
            if self.config == "aq":
                eoq_token_id = self.tokenizer.encode("</q>", add_special_tokens=False)
                assert len(eoq_token_id) == 1
                self.eoq_token_id = eoq_token_id[0]
            elif self.config == "qa":
                eoa_token_id = self.tokenizer.encode("</a>", add_special_tokens=False)
                assert len(eoa_token_id) == 1
                self.eoa_token_id = eoa_token_id[0]
            elif self.config == "qa2s":
                eoq_token_id = self.tokenizer.encode("</q>", add_special_tokens=False)
                eoa_token_id = self.tokenizer.encode("</a>", add_special_tokens=False)
                assert len(eoq_token_id) == len(eoa_token_id) == 1
                self.eoa_token_id = eoa_token_id[0]
                self.eoq_token_id = eoq_token_id[0]

    def _extract(self, token_ids, second: bool = False):
        def get_token_index(token_ids, start_index, token_id):
            return (token_ids[start_index:] == token_id).nonzero(as_tuple=True)[0]

        if self.config == "qg":
            start_index = 0
            if self.seq2seq:
                # in case of seq2seq models there will be an additional token at the beginning
                start_index = 1
            question_start_index = start_index
            eos_indices = get_token_index(
                token_ids, question_start_index, self.tokenizer.eos_token_id
            )
            if len(eos_indices) == 0:
                # can't extract question
                return None, None, None, None
            question_end_index = question_start_index + eos_indices[0]
            return question_start_index, question_end_index, None, None
        elif self.config == "q":
            if self.seq2seq:
                soq_indices = get_token_index(token_ids, 0, self.soq_token_id)
                if len(soq_indices) == 0:
                    # can't extract question
                    return None, None, None, None
                question_start_index = soq_indices[0] + 1
            else:
                question_start_index = 0
            eoq_indices = get_token_index(
                token_ids, question_start_index, self.eoq_token_id
            )
            if len(eoq_indices) == 0:
                # can't extract question
                return None, None, None, None
            question_end_index = question_start_index + eoq_indices[0]
            return question_start_index, question_end_index, None, None
        elif self.config == "aq":
            if self.seq2seq:
                soa_indices = get_token_index(token_ids, 0, self.soa_token_id)
                if len(soa_indices) == 0:
                    # can't extract answer
                    return None, None, None, None
                answer_start_index = soa_indices[0] + 1
            else:
                answer_start_index = 0
            eoa_indices = get_token_index(
                token_ids, answer_start_index, self.soq_token_id
            )
            if len(eoa_indices) == 0:
                # can't extract answer
                return None, None, None, None
            answer_end_index = answer_start_index + eoa_indices[0]
            question_start_index = answer_end_index + 1
            eoq_indices = get_token_index(
                token_ids, question_start_index, self.eoq_token_id
            )
            if len(eoq_indices) == 0:
                # can't extract question
                return None, None, None, None
            question_end_index = question_start_index + eoq_indices[0]
            return (
                question_start_index,
                question_end_index,
                answer_start_index,
                answer_end_index,
            )
        elif self.config == "qa":
            if self.seq2seq:
                soq_indices = get_token_index(token_ids, 0, self.soq_token_id)
                if len(soq_indices) == 0:
                    # can't extract question
                    return None, None, None, None
                question_start_index = soq_indices[0] + 1
            else:
                question_start_index = 0
            eoq_indices = get_token_index(
                token_ids, question_start_index, self.soa_token_id
            )
            if len(eoq_indices) == 0:
                # can't extract question
                return None, None, None, None
            question_end_index = question_start_index + eoq_indices[0]
            answer_start_index = question_end_index + 1
            eoa_indices = get_token_index(
                token_ids, answer_start_index, self.eoa_token_id
            )
            if len(eoa_indices) == 0:
                # can't extract answer
                return None, None, None, None
            answer_end_index = answer_start_index + eoa_indices[0]
            return (
                question_start_index,
                question_end_index,
                answer_start_index,
                answer_end_index,
            )
        elif self.config == "qa2s":
            if second:
                # extract answer only (second step)
                if self.seq2seq:
                    soa_indices = get_token_index(token_ids, 0, self.soa_token_id)
                    if len(soa_indices) == 0:
                        # can't extract question
                        return None, None, None, None
                    answer_start_index = soa_indices[0] + 1
                else:
                    raise NotImplementedError()
                    # answer_start_index = 0
                eoa_indices = get_token_index(
                    token_ids, answer_start_index, self.eoa_token_id
                )
                if len(eoa_indices) == 0:
                    # can't extract answer
                    return None, None, None, None
                answer_end_index = answer_start_index + eoa_indices[0]
                return None, None, answer_start_index, answer_end_index
            else:
                # extract question only (first step)
                if self.seq2seq:
                    soq_indices = get_token_index(token_ids, 0, self.soq_token_id)
                    if len(soq_indices) == 0:
                        # can't extract question
                        return None, None, None, None
                    question_start_index = soq_indices[0] + 1
                else:
                    raise NotImplementedError()
                    # question_start_index = 0
                eoq_indices = get_token_index(
                    token_ids, question_start_index, self.eoq_token_id
                )
                if len(eoq_indices) == 0:
                    # can't extract question
                    return None, None, None, None
                question_end_index = question_start_index + eoq_indices[0]
                return question_start_index, question_end_index, None, None
        raise NotImplementedError()

    def _compute_lm_score(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        offset: int,
        start_index: int,
        end_index: int,
        token_scores: torch.Tensor,
    ):
        # this is using the provided scores from the generate method
        # score = token_scores[start_index:end_index].gather(dim=1, index=gen_token_ids[offset+start_index:offset+end_index].unsqueeze(1)).sum().float()
        # this is using the model again similar to what they do in https://huggingface.co/transformers/perplexity.html to compute the perplexity
        with torch.no_grad():
            target_ids = output_ids.clone().unsqueeze(0)
            target_ids[:, : offset + start_index] = -100
            target_ids[:, offset + end_index :] = -100

            if self.seq2seq:
                target_ids = target_ids[..., 1:]
                decoder_input_ids = output_ids.clone().unsqueeze(0)[..., :-1]
                outputs = self.model(
                    input_ids, labels=target_ids, decoder_input_ids=decoder_input_ids
                )
            else:
                outputs = self.model(input_ids, labels=target_ids)
            # outputs[0] is the average negative log likelihood per token
            score = -1.0 * outputs[0].cpu().item() * (end_index - start_index)
        return score

    def _generate_and_extract(self, sample: Dict):
        """Generate output token ids from inputs and extract question and answer with scores by computing their spans in the decoded sequence"""
        # prepare inputs
        inputs = {
            "input_ids": torch.tensor(
                sample["input_ids"], device=self.args.device
            ).unsqueeze(0),
            "attention_mask": torch.tensor(
                sample["attention_mask"], device=self.args.device
            ).unsqueeze(0),
        }
        if "token_type_ids" in sample:
            inputs["token_type_ids"] = torch.tensor(
                sample["token_type_ids"], device=self.args.device
            ).unsqueeze(0)

        # set eos token
        if self.config == "qg":
            eos_token_id = self.tokenizer.eos_token_id
        else:
            if self.config == "q":
                eos_token_id = self.tokenizer.encode("</q>", add_special_tokens=False)
            elif self.config == "aq":
                if not self.seq2seq and self.custom_token_type_ids:
                    eos_token_id = self.tokenizer.encode(
                        "<q>", add_special_tokens=False
                    )
                else:
                    eos_token_id = self.tokenizer.encode(
                        "</q>", add_special_tokens=False
                    )
            elif self.config == "qa":
                if not self.seq2seq and self.custom_token_type_ids:
                    eos_token_id = self.tokenizer.encode(
                        "<a>", add_special_tokens=False
                    )
                else:
                    eos_token_id = self.tokenizer.encode(
                        "</a>", add_special_tokens=False
                    )
            elif self.config == "qa2s":
                if not self.seq2seq:  # and self.custom_token_type_ids:
                    raise NotImplementedError()
                    # eos_token_id = self.tokenizer.encode('<a>', add_special_tokens=False)
                else:
                    eos_token_id = self.tokenizer.encode(
                        "</q>", add_special_tokens=False
                    )
                    # eos token for second decoding step
                    eos_token_id_2 = self.tokenizer.encode(
                        "</a>", add_special_tokens=False
                    )
                    assert len(eos_token_id_2) == 1
                    eos_token_id_2 = eos_token_id_2[0]
            # if correctly added to the dictionary then those tokens won't be split
            assert len(eos_token_id) == 1
            eos_token_id = eos_token_id[0]

        # set bos token
        if self.seq2seq:
            if self.config == "qg":
                bos_token_id = self.tokenizer.bos_token_id
            else:
                if self.config == "aq":
                    bos_token_id = self.tokenizer.encode(
                        "<a>", add_special_tokens=False
                    )
                else:
                    if self.config == "qa2s":
                        # bos token for second decoding step
                        bos_token_id_2 = self.tokenizer.encode(
                            "<a>", add_special_tokens=False
                        )
                        assert len(bos_token_id_2) == 1
                        bos_token_id_2 = bos_token_id_2[0]

                    bos_token_id = self.tokenizer.encode(
                        "<q>", add_special_tokens=False
                    )
                # if correctly added to the dictionary then those tokens won't be split
                assert len(bos_token_id) == 1
                bos_token_id = bos_token_id[0]
        else:
            bos_token_id = None

        # parameters from Shakeri et al.
        if self.config in ["q", "qg"]:
            generation_output = self.model.generate(
                **inputs,
                max_length=(
                    self.max_gen_length
                    if self.model.config.is_encoder_decoder
                    else inputs["input_ids"].size(-1) + self.max_gen_length
                ),
                max_new_tokens=None,
                num_return_sequences=1,
                num_beams=10,
                do_sample=False,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=bos_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                forced_eos_token_id=eos_token_id,
            )
        elif self.config in ["aq", "qa", "qa2s"]:
            if self.config == "qa2s":
                assert (
                    inputs["input_ids"].size(-1) + self.max_gen_length
                    <= self.tokenizer.model_max_length
                )
            max_length = (
                self.max_gen_length
                if self.model.config.is_encoder_decoder
                else inputs["input_ids"].size(-1) + self.max_gen_length
            )
            generation_output = self.model.generate(
                **inputs,
                max_length=max_length,
                max_new_tokens=None,
                do_sample=True,
                top_k=20,
                top_p=0.95,
                num_return_sequences=10,
                num_beams=1,
                # early_stopping=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=bos_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                forced_eos_token_id=eos_token_id,
            )
            if not self.seq2seq and self.custom_token_type_ids:
                # two step decoding for models like GPT2 with correct token_type_ids set (1 for first generated sequence, 2 for second generated sequence)
                raise NotImplementedError()
                # replace generation_output

        # scores are before softmax hence we apply it as well as log (for summing the scores later)
        generation_scores = torch.stack(generation_output.scores, dim=0).log_softmax(
            dim=-1
        )

        if self.seq2seq:
            start_index = 0
        else:
            start_index = inputs["input_ids"].size(dim=1)

        questions, answers, scores = [], [], []

        for idx, generated_sequence in enumerate(generation_output.sequences):
            gen_sequence = generated_sequence[start_index:]
            question_start, question_end, answer_start, answer_end = self._extract(
                gen_sequence
            )
            if question_start == question_end == None:
                continue
            # make sure that indices are on CPU
            question_start, question_end = int(question_start), int(question_end)

            question = self.tokenizer.decode(
                gen_sequence[question_start:question_end],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            ).strip()
            if self.config == "qg":
                # in this config there are additional 'question:' and ':question' tokens added
                # those have to appear for a valid question but won't be considered as part of the question
                if question.startswith("question:") and question.endswith(":question"):
                    question = question[9:-9].strip()
                else:
                    continue
            # question cannot be empty or occur multiple times
            if question == "":
                continue
            if self.config in ["aq", "qa", "qa2s"]:
                # for qa2s we have to do a second decoding step
                if self.config == "qa2s":
                    # prepare input
                    if not self.seq2seq:
                        raise NotImplementedError()

                    inputs = {
                        "input_ids": torch.cat(
                            (
                                torch.tensor(
                                    sample["input_ids"], device=self.args.device
                                ),
                                torch.tensor(
                                    [self.soq_token_id], device=self.args.device
                                ),
                                gen_sequence[question_start:question_end],
                            ),
                            dim=0,
                        ).unsqueeze(0)
                    }
                    max_length = (
                        self.max_gen_length
                        if self.model.config.is_encoder_decoder
                        else inputs["input_ids"].size(-1) + self.max_gen_length
                    )

                    # do greedy decoding for answer
                    generation_output = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        max_new_tokens=None,
                        num_return_sequences=1,
                        num_beams=1,
                        do_sample=False,
                        early_stopping="never",
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        decoder_start_token_id=bos_token_id_2,
                        bos_token_id=bos_token_id_2,
                        eos_token_id=eos_token_id_2,
                        forced_eos_token_id=eos_token_id_2,
                    )

                    # get score (answer score is sufficient for qa2s)
                    generation_scores = torch.stack(
                        generation_output.scores, dim=0
                    ).log_softmax(dim=-1)

                    # update start_index
                    if self.seq2seq:
                        start_index_2 = 0
                    else:
                        start_index_2 = inputs["input_ids"].size(dim=1)

                    generated_sequence = generation_output.sequences[0]
                    gen_sequence = generated_sequence[start_index_2:]
                    # this time we're interested in the answer only
                    _, _, answer_start, answer_end = self._extract(
                        gen_sequence, second=True
                    )

                # for these configs we have to extract both question and answer
                if answer_start == answer_end == None:
                    continue

                # make sure that indices are on CPU
                answer_start, answer_end = int(answer_start), int(answer_end)

                answer = self.tokenizer.decode(
                    gen_sequence[answer_start:answer_end],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                ).strip()

                # make sure that answer is not empty and appears in context
                if answer == "" or answer not in sample["context"]:
                    continue
                answers.append(answer)
            questions.append(question)

            # extract LM scores for later use in filtering
            if self.config in ["qg", "q"]:
                # extract question score only for completeness
                # questions score
                score = self._compute_lm_score(
                    inputs["input_ids"] if self.seq2seq else generated_sequence,
                    generated_sequence,
                    start_index,
                    question_start,
                    question_end,
                    generation_scores[question_start:question_end, idx, :].squeeze(1),
                )
                scores.append(score)
            elif self.config == "aq":
                # extract answer & question score
                # answer score
                gen_scores_current_seq = generation_scores[:, idx, :].squeeze(1)
                score = self._compute_lm_score(
                    inputs["input_ids"] if self.seq2seq else generated_sequence,
                    generated_sequence,
                    start_index,
                    answer_start,
                    answer_end,
                    gen_scores_current_seq,
                )
                # question score
                score += self._compute_lm_score(
                    inputs["input_ids"] if self.seq2seq else generated_sequence,
                    generated_sequence,
                    start_index,
                    question_start,
                    question_end,
                    gen_scores_current_seq,
                )
                scores.append(score)
            elif self.config == "qa":
                # extract answer score
                # answer score
                score = self._compute_lm_score(
                    inputs["input_ids"] if self.seq2seq else generated_sequence,
                    generated_sequence,
                    start_index,
                    answer_start,
                    answer_end,
                    generation_scores[answer_start:answer_end, idx, :].squeeze(1),
                )
                scores.append(score)
            elif self.config == "qa2s":
                # extract answer score
                # answer score
                # NOTE fort qa2s we decode the answer greedily hence there is only one output and one score
                score = self._compute_lm_score(
                    inputs["input_ids"] if self.seq2seq else generated_sequence,
                    generated_sequence,
                    start_index,
                    answer_start,
                    answer_end,
                    generation_scores[answer_start:answer_end, 0, :].squeeze(1),
                )
                scores.append(score)
            assert scores

        return questions, answers, scores

    def predict(self, test_dataset: Dataset):
        # dataloader = self.get_test_dataloader(test_dataset)
        self.model.eval()

        predictions = []

        for sample in tqdm(test_dataset):
            questions, answers, scores = self._generate_and_extract(sample)
            if not questions:
                # no generated sample could be extracted
                continue

            if self.config in ["aq", "qa", "qa2s"]:
                # answer was predicted together with question
                gen_answers = []
                for answer in answers:
                    char_start = find_answer_span(sample["context"], answer)[0]
                    assert char_start >= 0
                    gen_answers.append({"answer_start": [char_start], "text": [answer]})
                answers = gen_answers
            elif self.config in ["qg", "q"]:
                # answer was predicted in previous step
                assert "answers" in sample
                answers = [sample["answers"]] * len(questions)
                # answers = cycle([sample['answers']])

            predictions.append(
                {
                    "id": sample["id"],
                    "scores": scores,
                    "context": sample["context"],
                    "questions": questions,
                    "answers": answers,
                }
            )

        return predictions
        # return Dataset.from_dict(dicts_to_feature_dict(predictions))

    def predict_to_file(self, test_dataset: Dataset, dir: str, size: int = 1000):
        """Predict `test_dataset` in shards and write to `dir`"""

        dataset_list = []
        num_shards = len(test_dataset) // size
        for i in range(num_shards):
            dataset_list.append(
                self.predict(test_dataset.shard(num_shards, i, contiguous=True))
            )


@dataclass
class LMFilter:
    num_keep: int

    def filter_lm_score(self, questions, answers, scores, num_keep, **kwargs):
        # `num_keep` best questions & answers
        indices = np.argsort(scores)[: -1 - num_keep : -1]
        return (
            [questions[idx] for idx in indices],
            [answers[idx] for idx in indices],
            [scores[idx] for idx in indices],
        )

    def __call__(self, data: Dict):
        # unpack samples and apply filtering
        return unpack_samples(data, filter_fn=partial(self.filter_lm_score, num_keep=5))


class RTFilter(RCTrainer):
    def __call__(self, data: Dict):
        # unpack samples first
        data = unpack_samples(data)
        # do not use the answer column for preparing the features in order to do inference only
        feats = data.map(
            prepare_rc_features,
            batched=True,
            keep_in_memory=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "question_column": "question",
                "answer_column": None,
                "question_max_length": self.question_max_length,
                "as_training_data": False,
                "with_labels": True,
            },
            num_proc=self.num_worker,
        )
        # apply fix for transformers trainer removing first dimension if batch-size is 1
        feats = feats.map(
            lambda x: {"transformers_fix": True},
            keep_in_memory=True,
            num_proc=self.num_worker,
        )
        # do prediction
        logits = self.predict(feats).predictions
        # extract spans from logits
        preds = rc_predict(
            logits,
            feats["id"],
            feats["id"],
            feats["context"],
            feats["context_offset"],
            feats["length"],
            feats["offset_mapping"],
            1,
            self.max_answer_length,
        )
        # for each sample there has to be a prediction
        assert len(data) == len(preds)
        # filter samples where the prediction matches the generated answer
        data = data.filter(
            lambda x: x["answers"]["text"][0]
            == preds[x["id"]].extract_answers(1)[1][0],
            num_proc=self.num_worker,
        )
        return data


class ActiveLearner:
    """A class which takes care of active learning and implements several confidence scores for generation models."""

    def __init__(
        self,
        output_dir,
        max_gen_length,
        rc_process_data_fn,
        gen_process_data_fn,
        gen_special_token_id_map,
        **trainer_kwargs,
    ):
        rc_trainer_kwargs, gen_trainer_kwargs, remaining_args = separate_map_on_prefix(
            trainer_kwargs, "rc_", "gen_"
        )
        assert not remaining_args

        # bos_token_id may be None for non-seq2seq models
        self.gen_special_token_id_map = gen_special_token_id_map

        # dir for storing drawn samples
        self.output_dir = output_dir

        # set some arguments for AL
        # we need best model loaded in the end so that we can use it for scoring the pool
        if rc_process_data_fn is not None:
            rc_trainer_kwargs.pop("train_dataset", None)
            rc_trainer_kwargs["args"].load_best_model_at_end = True
            rc_trainer_kwargs["args"].metric_for_best_model = "f1"
            rc_trainer_kwargs["args"].greater_is_better = True
        else:
            rc_trainer_kwargs = None
        if gen_process_data_fn is not None:
            max_gen_length = 200
            assert (
                max_gen_length is not None and max_gen_length != -1
            ), "`max_gen_length` cannot be None or -1 for AL"
            self.max_gen_length = max_gen_length
            gen_trainer_kwargs.pop("train_dataset", None)
            gen_trainer_kwargs["args"].load_best_model_at_end = True
            gen_trainer_kwargs["args"].metric_for_best_model = "loss"
            gen_trainer_kwargs["args"].greater_is_better = False
        else:
            gen_trainer_kwargs = None
        self.rc_trainer = None
        self.rc_process_data_fn = rc_process_data_fn
        self.rc_trainer_kwargs = rc_trainer_kwargs
        self.gen_trainer = None
        self.gen_process_data_fn = gen_process_data_fn
        self.gen_trainer_kwargs = gen_trainer_kwargs

    def create_gen_trainer(self, args: Dict = None, **kwargs):
        trainer_kwargs = self.gen_trainer_kwargs.copy()
        if args:
            trainer_kwargs["args"] = copy.deepcopy(trainer_kwargs["args"])
            for k, v in args.items():
                setattr(trainer_kwargs["args"], k, v)
        self.gen_trainer = Trainer(**trainer_kwargs, **kwargs)

    def create_rc_trainer(self, args: Dict = None, **kwargs):
        trainer_kwargs = self.rc_trainer_kwargs.copy()
        trainer_kwargs["args"] = copy.deepcopy(trainer_kwargs["args"])
        if args:
            for k, v in args.items():
                setattr(trainer_kwargs["args"], k, v)
        self.rc_trainer = RCTrainer(max_answer_length=None, **trainer_kwargs, **kwargs)

    def init_trainer(
        self, data_train=None, iteration: int = None, num_total_iterations: int = None
    ):
        # NOTE will also create new comet.ml experiment
        # set run name to reflect al round

        if iteration is not None:
            assert num_total_iterations is not None

        if self.gen_trainer_kwargs is not None:
            if data_train is not None:
                # process data for gen trainer
                gen_data_train = self.gen_process_data_fn(
                    data=data_train, with_labels=True, is_eval_data=False
                )
            else:
                gen_data_train = None
            if iteration is not None:
                args_dict = {
                    "run_name": f"{self.gen_trainer_kwargs['args'].run_name}_gen-round-{iteration+1}-of-{num_total_iterations}",
                    "output_dir": f"{self.gen_trainer_kwargs['args'].output_dir}_round-{iteration+1}-of-{num_total_iterations}",
                }
                # make sure to log in last iteration for averaging metrics over runs
                if iteration < num_total_iterations - 1:
                    for callback in self.gen_trainer_kwargs["callbacks"]:
                        if isinstance(callback, PyLoggerCallback):
                            callback.disable()
                else:
                    for callback in self.gen_trainer_kwargs["callbacks"]:
                        if isinstance(callback, PyLoggerCallback):
                            callback.enable()
            else:
                args_dict = None
            self.create_gen_trainer(train_dataset=gen_data_train, args=args_dict)
        if self.rc_trainer_kwargs is not None:
            if data_train is not None:
                # process data for rc trainer
                rc_data_train = self.rc_process_data_fn(
                    data=data_train, with_labels=True, is_eval_data=False
                )
            else:
                rc_data_train = None
            if iteration is not None:
                args_dict = {
                    "run_name": f"{self.rc_trainer_kwargs['args'].run_name}_rc-round-{iteration+1}-of-{num_total_iterations}",
                    "output_dir": f"{self.rc_trainer_kwargs['args'].output_dir}_round-{iteration+1}-of-{num_total_iterations}",
                }
                # make sure to log in last iteration for averaging metrics over runs
                if iteration < num_total_iterations - 1:
                    for callback in self.rc_trainer_kwargs["callbacks"]:
                        if isinstance(callback, PyLoggerCallback):
                            callback.disable()
                else:
                    for callback in self.rc_trainer_kwargs["callbacks"]:
                        if isinstance(callback, PyLoggerCallback):
                            callback.enable()
            else:
                args_dict = None
            self.create_rc_trainer(train_dataset=rc_data_train, args=args_dict)

    def train(self, iteration: int = None, num_total_iterations: int = None):
        # this call will re-initialize the model using the given checkpoint
        # NOTE this is the bahvior we want to have (i.e. re-train with all samples collected so far)
        if self.gen_trainer is not None:
            self.gen_trainer.evaluate()
            self.gen_trainer.train()
        if self.rc_trainer is not None:
            self.rc_trainer.evaluate()
            self.rc_trainer.train()

    def run_al(
        self,
        data: Dataset,
        rounds,
        num_samples,
        mode,
        sample_id_colum: str = "id",
        store_indices_and_scores: bool = True,
        store_samples: bool = True,
    ):
        # NOTE data contains twice as many instances per samples in case of qa2s but they have the same sample id and should return the same or a similar similarity score (but they have differently prepared inputs!)
        # create trainer
        self.init_trainer()

        # rc_dataset_train = rc_process_data_fn(rc_dataset_train, with_labels=True)
        # gen_dataset_train = gen_process_data_fn(gen_dataset_train, with_labels=True)

        # NOTE workaround for choosing only one data
        if mode in ["bs", "sp", "lex_sim", "rt", "sp+rt"]:
            pool = self.gen_process_data_fn(data, with_labels=False, is_eval_data=True)
            score_fn = self.score_gen
        else:
            pool = self.rc_process_data_fn(data, with_labels=False, is_eval_data=True)
            score_fn = self.score_rc

        # this will contain the sample ids (not the instance/table indices)
        selected_sample_indices = set()
        rounds = min(
            rounds,
            math.ceil(
                len(pool.flatten_indices().unique(sample_id_colum)) / num_samples
            ),
        )
        for i in range(rounds):
            logging.info(f"Running AL round {i + 1}/{rounds}")

            # evaluate samples from pool for confidence score
            pool_remaining = pool.filter(
                lambda x: x[sample_id_colum] not in selected_sample_indices,
                keep_in_memory=True,
            )
            if "cl_labels" in pool_remaining.column_names:
                pool_remaining = pool_remaining.filter(
                    lambda x: x["cl_labels"] != 1, keep_in_memory=True
                )
            if "qa2s_step" in pool_remaining.column_names:
                # for `qa2s` config discard samples which contain the question in the input (2nd step)
                pool_remaining = pool_remaining.filter(
                    lambda x: x["qa2s_step"] == 0, keep_in_memory=True
                )

            scored_instance_indices_sorted = score_fn(pool_remaining, mode, i)
            # we want to collect `num_samples` samples but one sample might have several instances hence we collect instances until we've got enough samples (or no data left)
            len_samples_drawn = len(selected_sample_indices)
            # currently we select the samples according to the max scores of their instances
            for idx, score in takewhile(
                lambda _: len(selected_sample_indices)
                < len_samples_drawn + num_samples,
                scored_instance_indices_sorted,
            ):
                selected_sample_indices.add(pool_remaining[idx][sample_id_colum])
            logger.info(
                "selected sample ids (%d): %s",
                len(selected_sample_indices),
                sorted(selected_sample_indices),
            )

            if store_indices_and_scores:
                # store sample ids in output dir
                run_dir = self.output_dir
                sample_ids_filename = os.path.join(
                    run_dir, f"al_samples_round_{i}.json"
                )
                logger.info(
                    f"Storing queried sample {sample_id_colum}s to {sample_ids_filename}"
                )
                with open(sample_ids_filename, "w") as f:
                    # sort list for debugging because set orders are random due to hash seed not being set
                    json.dump(
                        {
                            "iteration": i,
                            "num_rounds": rounds,
                            "num_select_samples": len(selected_sample_indices)
                            - len_samples_drawn,
                            "column": sample_id_colum,
                            "values": sorted(list(selected_sample_indices)),
                            "instances_and_scores": [
                                (index, pool_remaining[index][sample_id_colum], score)
                                for index, score in scored_instance_indices_sorted
                            ],
                        },
                        f,
                    )

            # add instances from selected samples to training data
            data_train = data.filter(
                lambda x: x[sample_id_colum] in selected_sample_indices,
                keep_in_memory=True,
            )

            if store_samples:
                # store samples in output dir
                run_dir = self.output_dir
                samples_dir = os.path.join(run_dir, f"al_samples_round_{i}")
                logger.info(f"Storing queried samples to {samples_dir}")
                data_train.save_to_disk(samples_dir)

            # create new trainer
            self.init_trainer(
                data_train=data_train, iteration=i, num_total_iterations=rounds
            )

            # train
            self.train(iteration=i, num_total_iterations=rounds)

    def generate_output(self, input_ids):
        assert (
            input_ids.size(0) == 1
        ), "We can only handle single instance batches currently"
        if self.gen_trainer.model.config.is_encoder_decoder:
            assert self.gen_special_token_id_map["bos_token_id"] is not None
        if (
            self.gen_special_token_id_map["bos_token_id_2"] is not None
            and self.gen_special_token_id_map["eos_token_id_2"] is not None
        ):
            # in this case we do two generation steps and the output from the first step has to fit into the model together with the input
            assert (
                input_ids.size(-1) + self.max_gen_length
                <= self.gen_trainer.tokenizer.model_max_length
            ), (
                input_ids.size(-1),
                self.max_gen_length,
                self.gen_trainer.tokenizer.model_max_length,
            )
        else:
            assert input_ids.size(-1) < self.gen_trainer.tokenizer.model_max_length
        # set max_length and ignore max_new_tokens as it counts differently
        generated_sequences = self.gen_trainer.model.generate(
            input_ids,
            max_length=(
                self.max_gen_length
                if self.gen_trainer.model.config.is_encoder_decoder
                else input_ids.size(-1) + self.max_gen_length
            ),
            max_new_tokens=None,
            num_return_sequences=1,
            num_beams=10,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=self.gen_trainer.tokenizer.pad_token_id,
            decoder_start_token_id=self.gen_special_token_id_map["bos_token_id"],
            bos_token_id=self.gen_special_token_id_map["bos_token_id"],
            eos_token_id=self.gen_special_token_id_map["eos_token_id"],
            forced_eos_token_id=None,
        ).sequences

        if self.gen_trainer.model.config.is_encoder_decoder:
            start_index = 0
        else:
            start_index = input_ids.size(-1)
        if (
            self.gen_special_token_id_map["bos_token_id_2"] is not None
            and self.gen_special_token_id_map["eos_token_id_2"] is not None
        ):
            if self.gen_trainer.model.config.is_encoder_decoder:
                input_ids = torch.cat(
                    (input_ids, generated_sequences[..., :-1]), dim=-1
                )
            else:
                input_ids = generated_sequences
            assert (
                input_ids.size(-1) <= self.gen_trainer.tokenizer.model_max_length
            ), f"Input size ({input_ids.size(-1)}) has to be smaller or equal than model max length ({self.gen_trainer.tokenizer.model_max_length}); generated sequence length in previous step is {generated_sequences.size(-1)} with max generation length set to {self.max_gen_length}"

            generated_sequences = generated_sequences[:, start_index:]
            if not self.gen_trainer.model.config.is_encoder_decoder:
                start_index = input_ids.size(-1)
            generated_sequences = (
                generated_sequences,
                self.gen_trainer.model.generate(
                    input_ids,
                    max_length=(
                        self.max_gen_length
                        if self.gen_trainer.model.config.is_encoder_decoder
                        else input_ids.size(-1) + self.max_gen_length
                    ),
                    max_new_tokens=None,
                    num_return_sequences=1,
                    num_beams=10,
                    return_dict_in_generate=True,
                    output_scores=False,
                    pad_token_id=self.gen_trainer.tokenizer.pad_token_id,
                    decoder_start_token_id=self.gen_special_token_id_map[
                        "bos_token_id_2"
                    ],
                    bos_token_id=self.gen_special_token_id_map["bos_token_id_2"],
                    eos_token_id=self.gen_special_token_id_map["eos_token_id_2"],
                    forced_eos_token_id=None,
                ).sequences[:, start_index:],
            )
        else:
            generated_sequences = (generated_sequences[:, start_index:],)

        return input_ids, generated_sequences

    def lm_score(self, input_ids, output_ids, lengths=None):
        """input_ids and outputs_ids are handled as 1D inputs."""
        if lengths is None:
            assert (
                input_ids.size(0) == 1
            ), "For inputs with batch size > 1 you have to pass the `lengths` parameter"
            lengths = input_ids.size(1)

        with torch.no_grad():
            if self.gen_trainer.model.config.is_encoder_decoder:
                label_ids = output_ids[:, 1:]
                decoder_input_ids = output_ids[:, :-1]
                outputs = self.gen_trainer.model(
                    input_ids.contiguous(),
                    labels=label_ids.contiguous(),
                    decoder_input_ids=decoder_input_ids.contiguous(),
                )
            else:
                raise NotImplementedError(
                    "lm score computation for non-seq2seq models is currently not implemented correctly"
                )
            score = -1.0 * outputs[0].cpu().item() * lengths
        return score

    def beam_score(self, sample, return_generated_sequences: bool = False):
        """The input ids and output ids (including bos token)"""

        def length_penalty(seq_len, exp):
            return ((5 + seq_len) / 6) ** exp

        input_ids = torch.tensor(
            [sample["input_ids"]], device=self.gen_trainer.args.device
        )
        input_ids, output_ids = self.generate_output(input_ids)

        # we omit the log in the computation here since it's a monotonic function
        score = self.lm_score(input_ids, output_ids[-1]) / length_penalty(
            (
                input_ids.numel()
                if isinstance(input_ids, torch.Tensor)
                else len(input_ids)
            ),
            0.6,
        )
        if return_generated_sequences:
            return score, tuple(_output_ids[0] for _output_ids in output_ids)
        else:
            return score

    def sentence_probability(
        self, sample, num_models: int = 10, return_generated_sequences: bool = False
    ):
        """The input ids and output ids (including bos token)"""
        # generate output once and use for computing log-likelihood with different models (by activating dropout)
        self.gen_trainer.model.eval()
        input_ids = torch.tensor(
            [sample["input_ids"]], device=self.gen_trainer.args.device
        )
        input_ids, output_ids = self.generate_output(input_ids)

        # activate dropout
        self.gen_trainer.model.train()

        # we omit the log in the computation here since it's a monotonic function
        # also we omit the length penalty since this is the D-TP from Fomicheva using averaged scores over samples and tokens
        score = self.lm_score(
            input_ids.expand(num_models, -1),
            output_ids[-1].expand(num_models, -1),
            lengths=1,
        )
        if return_generated_sequences:
            return score, tuple(_output_ids[0] for _output_ids in output_ids)
        else:
            return score

    sim_fn = load_metric("meteor")

    def lex_similarity(
        self, sample, num_models: int = 10, return_generated_sequences: bool = False
    ):
        # generate `num_models` hypotheses with dropout active
        self.gen_trainer.model.train()
        input_ids = torch.tensor(
            [sample["input_ids"]], device=self.gen_trainer.args.device
        )
        hyps = [
            self.gen_trainer.tokenizer.decode(self.generate_output(input_ids)[1][-1][0])
            for _ in range(num_models)
        ]
        # compute the cartesian product of the list of decoded hypotheses with itself and remove tuples with same hyps (i.e. where i==j)
        hyps_pairwise = list(itertools.product(hyps, repeat=2))
        step = (num_models**2 - 1) // (num_models - 1)
        for i in range(num_models):
            del hyps_pairwise[i * step - i]
        predictions_1, predictions_2 = zip(*hyps_pairwise)
        score = self.sim_fn.compute(
            predictions=predictions_1, references=predictions_2
        )["meteor"]
        if return_generated_sequences:
            raise NotImplementedError(
                "Computation of returned generated sequence hasn't been implemented yet."
            )
            return score, tuple(_output_ids[0] for _output_ids in output_ids)
        else:
            return score

    def bald(self, sample: dict, num_models: int = 10):
        # compute bald score (maximize the information gained about the model parameters) with dropout active
        # this is an approximation where we keep start and end probability separate since we have a classification over tokens for start and end probability
        # ideally one would consider the cartesian product of start and end tokens
        self.rc_trainer.model.train()

        with torch.no_grad():
            # don't use trainer for prediction as we have a dict instead of Dataset
            # by expanding the batch dimension we effectly do `num_models`forward passes
            logits: torch.Tensor = self.rc_trainer.model(
                input_ids=torch.tensor(
                    [sample["input_ids"]], device=self.rc_trainer.args.device
                ).expand(num_models, -1),
                token_type_ids=torch.tensor(
                    [sample["token_type_ids"]], device=self.rc_trainer.args.device
                ).expand(num_models, -1),
                attention_mask=torch.tensor(
                    [sample["attention_mask"]], device=self.rc_trainer.args.device
                ).expand(num_models, -1),
            )
            # BALD from http://arxiv.org/abs/1703.02910
            mean_over_forward_passes = logits.mean(dim=0)
            score = -(mean_over_forward_passes * mean_over_forward_passes.log()).sum(
                dim=0
            ) + (logits * logits.log()).sum(dim=1).mean(dim=0)
            score = -score.sum().item()

        return score

    def score_rc(self, data: Dataset, strategy: str, iteration: int):
        # store current model mode so that we can reset it later
        train_mode = self.rc_trainer.model.training

        if strategy == "bald":
            scores = self.score_dataset(data, self.bald)
        else:
            raise NotImplementedError(
                f"There is no such AL query strategy '{strategy}' implemented"
            )

        # reset model mode
        self.rc_trainer.model.train(train_mode)

        indices_sorted = np.argsort(scores)
        return list(zip(indices_sorted.tolist(), np.asarray(scores)[indices_sorted]))

    def score_gen(self, data: Dataset, strategy: str, iteration: int):
        # store current model mode so that we can reset it later
        train_mode = self.gen_trainer.model.training

        if strategy == "bs":
            scores = self.score_dataset(data, self.beam_score)
        elif strategy == "sp":
            scores = self.score_dataset(data, self.sentence_probability)
        elif strategy == "lex_sim":
            scores = self.score_dataset(data, self.lex_similarity)
        elif strategy in ["rt", "sp+rt"]:
            scores_gen, generated_sequences = zip(
                *self.score_dataset(
                    data, self.sentence_probability, return_generated_sequences=True
                )
            )
            scores_rc = []
            assert len(data) == len(scores_gen) == len(generated_sequences)
            for sample, (question_token_ids, answer_token_ids) in zip(
                data, generated_sequences
            ):
                # extract question and answer
                question, answer = self.gen_trainer.tokenizer.convert_tokens_to_string(
                    self.gen_trainer.tokenizer.convert_ids_to_tokens(
                        question_token_ids, skip_special_tokens=True
                    )
                ), self.gen_trainer.tokenizer.convert_tokens_to_string(
                    self.gen_trainer.tokenizer.convert_ids_to_tokens(
                        answer_token_ids, skip_special_tokens=True
                    )
                )
                if answer not in sample["context"]:
                    # cannot apply RT model since we cannot infer labels
                    scores_rc.append(0.0)
                else:
                    # run prediction and compute score
                    char_start = find_answer_span(sample["context"], answer)[0]
                    assert char_start >= 0
                    predicted_answer = self.rc_trainer.preprocess_and_predict_sample(
                        Dataset.from_dict(
                            {
                                "id": [sample["id"]],
                                "context": [sample["context"]],
                                "question": [question],
                                "answers": [
                                    {
                                        "answer_start": [char_start],
                                        "answer_text": [answer],
                                    }
                                ],
                            }
                        )
                    )
                    scores_rc.append(
                        RCMetric.max_over_ground_truths(
                            RCMetric.f1, predicted_answer, [answer]
                        )
                    )
            if strategy == "rt":
                scores = scores_rc
            else:
                scores_gen = np.square(np.exp(np.multiply(scores_gen, 4)))
                scores = np.add(scores_gen, scores_rc)
        else:
            raise NotImplementedError(
                f"There is no such AL query strategy '{strategy}' implemented"
            )

        # dataloader = self.get_test_dataloader(test_dataset)

        # reset model mode
        self.gen_trainer.model.train(train_mode)

        indices_sorted = np.argsort(scores)
        return list(zip(indices_sorted.tolist(), np.asarray(scores)[indices_sorted]))

    def score_dataset(self, data, score_fn, **kwargs):
        scores = []
        # this iterates row by row, i.e., row 0, 1, 2, ... hence order of list `scores` is important
        for sample in tqdm(data):
            scores.append(score_fn(sample, **kwargs))
        return scores
