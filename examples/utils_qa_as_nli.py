# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" NLI processors and helpers """

import logging
import os
import pandas as pd

import tqdm
import json
import copy
import csv
import sys

from typing import List, Tuple, Dict
import torch
from allennlp.training.metrics import Average
from allennlp.training.metrics import Metric

logger = logging.getLogger(__name__)


class SingleChoiceInputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, premise, hypothesis, label):
        self.guid = guid
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class MultipleChoiceInputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, example_id, premise, options, label):
        self.example_id = example_id
        self.premise = premise
        self.options = options
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemanticFragmentsInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a SemanticFragmentsInputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class SemanticFragmentsInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class SingelChoiceInputFeatures(object):
    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class MultipleChoiceInputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        } for input_ids, input_mask, segment_ids in choices_features]
        self.label = label


class DataProcessor():
    """Base class for data converters for QA as NLI tasks."""

    def get_train_examples(self,
                           data_dir,
                           hypothesis_type,
                           subset=None,
                           use_static_passage=False):
        """See base class."""

        if subset in ['rule', 'neural']:
            return self._create_examples(
                pd.read_json(os.path.join(data_dir, "train.json")).query(
                    '_'.join(['subset', subset])), hypothesis_type,
                use_static_passage)
        elif subset is None:
            return self._create_examples(
                pd.read_json(os.path.join(data_dir, "train.json")),
                hypothesis_type, use_static_passage)
        else:
            raise ValueError("Invalid subset flag")

    def get_dev_examples(self,
                         data_dir,
                         hypothesis_type,
                         subset=None,
                         use_static_passage=False):
        """See base class."""

        if subset in ['rule', 'neural']:
            return self._create_examples(
                pd.read_json(os.path.join(data_dir, "dev.json")).query(
                    '_'.join(['subset', subset])), hypothesis_type,
                use_static_passage)
        elif subset is None:
            return self._create_examples(
                pd.read_json(os.path.join(data_dir, "dev.json")),
                hypothesis_type, use_static_passage)
        else:
            raise ValueError("Invalid subset flag")

    def get_test_examples(self,
                          data_dir,
                          hypothesis_type,
                          subset=None,
                          use_static_passage=False):
        """See base class."""

        if subset in ['rule', 'neural']:
            return self._create_examples(
                pd.read_json(os.path.join(data_dir, "test.json")).query(
                    '_'.join(['subset', subset])), hypothesis_type,
                use_static_passage)
        else:
            return self._create_examples(
                pd.read_json(os.path.join(data_dir, "test.json")),
                hypothesis_type, use_static_passage)

    def _create_examples(self, data, hypothesis_type):
        """Create a collection of `InputExample`s from the data"""
        raise NotImplementedError()

    def get_labels(self, num_labels):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []

            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)

            return lines


class SemanticFragmentsProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self,
                           data_dir,
                           hypothesis_type=None,
                           subset=None,
                           use_static_passage=False):
        """See base class."""

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "challenge_train.tsv")),
            "train")

    def get_dev_examples(self,
                         data_dir,
                         hypothesis_type=None,
                         subset=None,
                         use_static_passage=False):
        """See base class."""

        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "challenge_dev.tsv")), "dev")

    def get_labels(self, num_labels=None):
        """See base class."""

        return ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            guid = line[0]  # "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                SemanticFragmentsInputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class SingleChoiceProcessor(DataProcessor):
    """Processor for the RACE converted to NLI data set."""

    def get_labels(self, num_labels=None):
        """See base class."""

        return [False, True]

    def _create_examples(self, data, hypothesis_type, static=False):
        """Creates examples for the training and dev sets."""
        examples = []
        hyp_column = "_".join([
            "hypothesis", hypothesis_type
        ]) if hypothesis_type is not None else "hypothesis"

        if static:
            logger.info("Using Static Premise!")

        for (i, row) in data.iterrows():
            guid = row['id']
            premise = row['premise'] if not static else row['premise_static']
            hypothesis = row[hyp_column]
            label = row['label']
            examples.append(
                SingleChoiceInputExample(
                    guid=guid,
                    premise=premise,
                    hypothesis=hypothesis,
                    label=label))

        return examples


class MultipleChoiceProcessor(DataProcessor):
    """Processor for the RACE converted to NLI data set."""

    def get_labels(self, num_labels=4):
        """See base class."""

        return range(num_labels)

    def _create_examples(self, data, hypothesis_type, static=False):
        """Creates examples for the training and dev sets."""
        examples = []
        hyp_column = "_".join(["hypothesis", hypothesis_type])

        if static:
            logger.info("Using Static Premise!")

        for (i, row) in data.iterrows():
            example_id = row['id']
            premise = row['premise'] if not static else row['premise_static']
            hypothesis_options = row[hyp_column]
            label = row['label']
            examples.append(
                MultipleChoiceInputExample(
                    example_id=example_id,
                    premise=premise,
                    options=hypothesis_options,
                    label=label))

        return examples


def semantic_fragments_convert_examples_to_features(
        examples,
        tokenizer,
        num_choices,
        max_length=512,
        label_list=None,
        task=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        no_passage=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"

            if len(tokens_a) > max_length - 2:
                tokens_a = tokens_a[:(max_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info(
                "input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            SemanticFragmentsInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))

    return features


def single_choice_convert_examples_to_features(
        examples,
        tokenizer,
        num_choices,
        max_length=512,
        label_list=None,
        task=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        no_passage=False,
):
    """
    Loads a data file into a list of ``NLIInputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: NLI task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``NLIInputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = processors[task]()

        if label_list is None:
            label_list = processor.get_labels(num_choices)
            logger.info("Using label list %s for task %s" % (label_list, task))

        if output_mode is None:
            output_mode = output_modes[task]
            logger.info(
                "Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len(examples)))

        inputs = tokenizer.encode_plus(
            example.hypothesis,
            add_special_tokens=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            max_length=max_length) if no_passage else tokenizer.encode_plus(
                example.premise,
                example.hypothesis,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                max_length=max_length)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join(
                [str(x) for x in inputs['input_ids']]))
            logger.info("attention_mask: %s" % " ".join(
                [str(x) for x in inputs['attention_mask']]))
            logger.info("token_type_ids: %s" % " ".join(
                [str(x) for x in inputs['token_type_ids']]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(SingelChoiceInputFeatures(**inputs, label=label))

    return features


def multiple_choice_convert_examples_to_features(
        examples,
        tokenizer,
        num_choices,
        max_length=512,
        label_list=None,
        task=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        no_passage=False,
):
    """
    Loads a data file into a list of ``MultipleChoiceInputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: NLI task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``MultipleChoiceInputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = processors[task]()

        if label_list is None:
            label_list = processor.get_labels(num_choices)
            logger.info("Using label list %s for task %s" % (label_list, task))

        if output_mode is None:
            output_mode = output_modes[task]
            logger.info(
                "Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in tqdm.tqdm(
            enumerate(examples), desc="convert examples to features"):

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        assert (len(example.options) == num_choices)

        for option in example.options:
            text_a = example.premise
            text_b = option

            inputs = tokenizer.encode_plus(
                text_b,
                add_special_tokens=True,
                pad_to_max_length=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                max_length=max_length,
            ) if no_passage else tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                pad_to_max_length=True,
                max_length=max_length,
                return_token_type_ids=True,
                return_attention_mask=True)

            if "num_truncated_tokens" in inputs and inputs[
                    "num_truncated_tokens"] > 0:
                logger.warning(
                    f"Attention! you are cropping {inputs['num_trunvated_tokens']} tokens for {example.example_id} "
                )

            input_ids, token_type_ids = inputs["input_ids"], inputs[
                "token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = inputs['attention_mask']

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask,
                                     token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.example_id))

            for choice_idx, (input_ids, attention_mask,
                             token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(
                    map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(
                    map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(
                    map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(
            MultipleChoiceInputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            ))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)

        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class F1WithThreshold(Metric):
    def __init__(self, flip_sign: bool = False) -> None:
        """
        Keep flip_sign=False if higher score => positive label
        """
        super().__init__()
        self._scores: List = []
        self._labels: List = []
        self.flip_sign = flip_sign
        self._threshold = None

    def reset(self) -> None:
        self._scores = []
        self._labels = []

    def __call__(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        if len(scores.shape) != 1:
            raise ValueError("Scores should be 1D")

        if len(labels.shape) != 1:
            raise ValueError("Labesl should be 1D")

        if scores.shape != labels.shape:
            raise ValueError("Shape of score should be same as labels")
        temp_scores = scores.detach().cpu()

        if self.flip_sign:
            temp_scores = -1 * temp_scores
        current_scores = temp_scores.tolist()
        self._scores.extend(current_scores)
        current_labels = labels.detach().cpu().tolist()
        self._labels.extend(current_labels)

    def compute_best_threshold_and_f1(
            self) -> Tuple[float, float, float, float]:
        # Assumes that lower scores have to be classified as pos

        total_pos = sum(self._labels)
        sorted_scores_and_labels = sorted(zip(self._scores, self._labels))
        true_pos = 0.0
        false_pos = 0.0
        best_thresh = 0.0
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0

        for score, label in sorted_scores_and_labels:
            true_pos += label
            false_pos += (1.0 - label)
            precision = true_pos / (true_pos + false_pos + 1e-8)
            recall = true_pos / total_pos
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_thresh = score
                best_precision = precision
                best_recall = recall
                best_f1 = f1

        if self.flip_sign:
            best_thresh = -1 * best_thresh
        self._sorted_scores_and_labels = sorted_scores_and_labels
        self._threshold = best_thresh

        return best_thresh, best_f1, best_precision, best_recall

    def get_metric(self, reset: bool) -> Dict:
        # this is an expensive operation,
        # lets do it only once when reset is true

        if reset:
            thresh, f1, precision, recall = self.compute_best_threshold_and_f1(
            )
            self.reset()
        else:
            thresh, f1, precision, recall = (0, 0, 0, 0)

        return precision, recall, f1, thresh


processors = {
    "single_choice": SingleChoiceProcessor,
    "multiple_choice": MultipleChoiceProcessor,
    "semantic_fragments": SemanticFragmentsProcessor,
}

output_modes = {
    "single_choice": "classification",
    "multiple_choice": "classification",
    "semantic_fragments": "classification",
}

convert_examples_to_features = {
    "single_choice": single_choice_convert_examples_to_features,
    "multiple_choice": multiple_choice_convert_examples_to_features,
    "semantic_fragments": semantic_fragments_convert_examples_to_features,
}
