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
import copy
import tqdm
import json

logger = logging.getLogger(__name__)


class NLIInputExample(object):
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


class QA2NLIInputExample(object):
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


class NLIInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
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


class QA2NLIInputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        } for input_ids, input_mask, segment_ids in choices_features]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""

        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]

        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def nli_convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
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
        processor = nli_processors[task]()

        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))

        if output_mode is None:
            output_mode = nli_output_modes[task]
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

        features.append(NLIInputFeatures(**inputs, label=label))

    return features


def qa2nli_convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        no_passage=False,
):
    """
    Loads a data file into a list of ``QA2NLIInputFeatures``

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
        a list of task-specific ``QA2NLIInputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = nli_processors[task]()

        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))

        if output_mode is None:
            output_mode = nli_output_modes[task]
            logger.info(
                "Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in tqdm.tqdm(
            enumerate(examples), desc="convert examples to features"):

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []

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
            logger.info("race_id: {}".format(example.example_id))

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
            QA2NLIInputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            ))

    return features


class NLIProcessor(DataProcessor):
    """Processor for the RACE converted to NLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""

        return [False, True]

    def _create_examples(self, data, type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, row) in data.iterrows():
            guid = row['id']
            premise = row['premise']
            hypothesis = row['hypothesis']
            label = row['label']
            examples.append(
                NLIInputExample(
                    guid=guid,
                    premise=premise,
                    hypothesis=hypothesis,
                    label=label))

        return examples


class QA2NLIProcessor(DataProcessor):
    """Processor for the RACE converted to NLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            pd.read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""

        return [0, 1, 2, 3]

    def _create_examples(self, data, type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, row) in data.iterrows():
            example_id = row['id']
            premise = row['premise']
            options = row['hypothesis_options']
            label = row['label']
            examples.append(
                QA2NLIInputExample(
                    example_id=example_id,
                    premise=premise,
                    options=options,
                    label=label))

        return examples


class QA2NLIProcessor3class(QA2NLIProcessor):
    def get_labels(self):
        return [0, 1, 2]


nli_tasks_num_labels = {"nli": 2, "race2nli": 4, "dream2nli": 3}

nli_processors = {
    "nli": NLIProcessor,
    "race2nli": QA2NLIProcessor,
    "dream2nli": QA2NLIProcessor3class
}

nli_output_modes = {
    "nli": "classification",
    "race2nli": "classification",
    "dream2nli": "classification"
}

convert_examples_to_features = {
    "nli": nli_convert_examples_to_features,
    "race2nli": qa2nli_convert_examples_to_features,
    "dream2nli": qa2nli_convert_examples_to_features
}
