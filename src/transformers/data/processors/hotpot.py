import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import random
from tqdm import tqdm

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_bert import whitespace_tokenize
from .utils import DataProcessor


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def hotpot_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training, nsp=1.0):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.lower().find(cleaned_answer_text.lower()) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
                if span_is_impossible and is_training and nsp < random.random():
                    continue
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            HotpotFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
            )
        )
    return features


def hotpot_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def hotpot_convert_examples_to_features_without_process(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
                                                      is_training, return_dataset=False, threads=0, nsp=1.0, max_sentence_num=10):
    logger.info('without multi-processing')
    unique_id = 1000000000
    features = []
    sentence_label_dict = {'0': 0, '1': 1}
    level_label_dict = {'easy': 0, 'medium': 1, 'hard': 2}
    type_label_dict = {'bridge': 0, 'comparison': 1}
    yes_no_label_dict = {'text':0, 'yes': 1, 'no': 2}
    yes_no = {}
    for (example_index, example) in tqdm(enumerate(examples)):
        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position
            end_position = example.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.lower().find(cleaned_answer_text.lower()) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue
        answers = example.answers
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )
        sentence_token_indices = []
        sentence_tok_indices = []

        sentence_labels = []
        sentence_titles = []
        type_label = None
        level_label = None
        if answers:

            answer = answers[0]
            type_label = type_label_dict[answer['type']]
            yes_no_label = yes_no_label_dict[answer['yes_no_answer']]
            level_label = level_label_dict[answer['level']]
            sentence_indices = answer['sentence_indices']
            sentence_indices_new = []
            doc_context = " ".join(example.doc_tokens)
            sentence_char_start = 0
            for sentence_index, sentence_indice in enumerate(sentence_indices):
                sentence_char_end = sentence_indice[1]
                if not (sentence_char_end < len(doc_context) and doc_context[sentence_char_end] == ' '):
                    sentence_char_end = sentence_char_end - 1
                if sentence_char_start > sentence_char_end:
                    sentence_char_end = sentence_char_start
                sentence_indices_new.append([sentence_char_start, sentence_char_end])
                sentence_char_start = sentence_char_end + 1
            sentence_indices = sentence_indices_new[:]
            sentence_labels = [sentence_label_dict[x] for x in answer['sentence_labels']]
            sentence_titles = answer['sentence_titles']
            sentence_token_indices = [[example.char_to_word_offset[x[0]], example.char_to_word_offset[x[1]]] for x in sentence_indices]
            for x in sentence_token_indices:
                if x[1] < len(orig_to_tok_index) - 1:
                    sentence_tok_end = orig_to_tok_index[x[1] + 1] - 1
                else:
                    sentence_tok_end = len(all_doc_tokens) - 1
                if x[0] < len(orig_to_tok_index) - 1:
                    sentence_tok_start = orig_to_tok_index[x[0] + 1] - 1
                else:
                    sentence_tok_start = len(all_doc_tokens) - 1

                sentence_tok_indices.append([sentence_tok_start, sentence_tok_end])

        spans = []

        truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
        sequence_added_tokens = (
            tokenizer.max_len - tokenizer.max_len_single_sentence + 1
            if "roberta" in str(type(tokenizer))
            else tokenizer.max_len - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context
        span_indices = [[span['start'], span['start'] + span['length'] - 1, span["truncated_query_with_special_tokens_length"]] for span in spans]
        span_sentence_indices = []
        span_sentence_labels = []
        span_sentence_titles = []
        span_sentence_is_max = []
        if sentence_labels:
            for span_indice in span_indices:
                indices_temp = []
                labels_temp = []
                titles_temp = []
                max_temp = []
                for sentence_index, sentence_tok_indice in enumerate(sentence_tok_indices):
                    if sentence_tok_indice[0] < span_indice[0] and sentence_tok_indice[1] >= span_indice[0]:
                        if sentence_tok_indice[1] - span_indice[0] > (sentence_tok_indice[1] - sentence_tok_indice[0]) // 2:
                                indices_temp.append([span_indice[0], sentence_tok_indice[1]])
                                labels_temp.append(sentence_labels[sentence_index])
                                titles_temp.append(sentence_titles[sentence_index])
                                max_temp.append(True)
                        else:
                            indices_temp.append([span_indice[0], sentence_tok_indice[1]])
                            labels_temp.append(0)
                            titles_temp.append(sentence_titles[sentence_index])
                            max_temp.append(False)
                    elif sentence_tok_indice[0] >=span_indice[0] and sentence_tok_indice[1] <= span_indice[1]:
                        indices_temp.append(sentence_tok_indice)
                        labels_temp.append(sentence_labels[sentence_index])
                        titles_temp.append(sentence_titles[sentence_index])
                        max_temp.append(True)
                    else:
                        if sentence_tok_indice[0] <= span_indice[1] and sentence_tok_indice[1] > span_indice[1]:
                            if span_indice[1] - sentence_tok_indice[0] > (sentence_tok_indice[1] - sentence_tok_indice[0]) // 2:
                                indices_temp.append([sentence_tok_indice[0], span_indice[1]])
                                labels_temp.append(sentence_labels[sentence_index])
                                titles_temp.append(sentence_titles[sentence_index])
                                max_temp.append(True)
                            else:
                                indices_temp.append([sentence_tok_indice[0], span_indice[1]])
                                labels_temp.append(0)
                                titles_temp.append(sentence_titles[sentence_index])
                                max_temp.append(False)
                span_sentence_indices.append(indices_temp)
                span_sentence_titles.append(titles_temp)
                span_sentence_labels.append(labels_temp)
                span_sentence_is_max.append(max_temp)
        span_sentence_new_indices = []
        for span_index, span_indice in enumerate(span_indices):
            span_sentence_new_indice = []
            span_sentence_indice = span_sentence_indices[span_index]
            span_query_length = span_indice[2]
            span_start = span_indice[0]
            for span_sentence in span_sentence_indice:
                sentence_tok_start = span_sentence[0] - span_start + span_query_length
                sentence_tok_end = span_sentence[1] - span_start + span_query_length
                try:
                    assert sentence_tok_start <= max_seq_length
                    assert sentence_tok_end <= max_seq_length
                except:
                    raise ValueError('sentence tok end > max_seq_length!')
                if sentence_tok_start > sentence_tok_end:
                    print('sentence start > sentence end!')
                span_sentence_new_indice.append([sentence_tok_start, sentence_tok_end])
            span_sentence_new_indices.append(span_sentence_new_indice[:])


        for span_index, span in enumerate(spans):
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span["token_type_ids"])

            p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                    if is_training and span_is_impossible and features and nsp < random.random():
                        continue
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                HotpotFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=example_index,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                    unique_id=unique_id,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    type_label=type_label,
                    level_label=level_label,
                    yes_no_label=yes_no_label,
                    span_sentence_labels=span_sentence_labels[span_index],
                    span_sentence_is_max=span_sentence_is_max[span_index],
                    span_sentence_titles=span_sentence_titles[span_index],
                    span_sentence_indices=span_sentence_new_indices[span_index],
                    sentence_tok_indices=sentence_tok_indices,
                    span_sentence_indices_in_paragraph=span_sentence_indices,
                    sentence_token_indices=sentence_token_indices
                )
            )
            unique_id += 1


    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        if is_training:
            max_sentence_num = min(max_sentence_num, max([len(f.span_sentence_indices) for f in features]))
        else:
            max_sentence_num = max([len(f.span_sentence_indices) for f in features])
            if max_sentence_num > 30:
                logging.info('In evaluating, max sentence num is very big: {}'.format(max_sentence_num))
        for f in features:
            pad_sentence_num = max_sentence_num - len(f.span_sentence_indices)
            if pad_sentence_num > 0:
                f.span_sentence_indices = f.span_sentence_indices + [[-1, -1] for _ in range(pad_sentence_num)]
                f.span_sentence_labels = f.span_sentence_labels + [0 for _ in range(pad_sentence_num)]
            else:
                f.span_sentence_indices = f.span_sentence_indices[:max_sentence_num]
                f.span_sentence_labels = f.span_sentence_labels[:max_sentence_num]
            assert len(f.span_sentence_indices) == len(f.span_sentence_labels)
        all_sentence_indices = torch.tensor([f.span_sentence_indices for f in features], dtype=torch.long)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask, all_sentence_indices
            )

        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_sentence_labels = torch.tensor([f.span_sentence_labels for f in features], dtype=torch.long)
            all_type_labels = torch.tensor([f.type_label for f in features], dtype=torch.long)
            all_yes_no_labels = torch.tensor([f.yes_no_label for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_sentence_indices,
                all_sentence_labels,
                all_type_labels,
                all_yes_no_labels,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {"start_position": tf.int64, "end_position": tf.int64, "cls_index": tf.int64, "p_mask": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                },
            ),
        )

    return features

def hotpot_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1, nsp=1.0
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.hotpot.HotpotExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.hotpot.HotpotFeatures`

    Example::

        processor = HopotV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = hotpot_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=hotpot_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            hotpot_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
            nsp=nsp,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert hotpot examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {"start_position": tf.int64, "end_position": tf.int64, "cls_index": tf.int64, "p_mask": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                },
            ),
        )

    return features


class HotpotProcessor(DataProcessor):
    """
    Processor for the hotpot data set.
    Overriden by HotpotV1Processor and HopotV2Processor, used by the version 1.1 and version 2.0 of hotpot, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return HotpotExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.hotpot.HotpotExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("hotpot")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of HotpotExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("hotpot")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for hotpot versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("HotpotProcessor should be instantiated via HotpotV1Processor or HopotV2Processor")
        logging.info('train file is :{}'.format(os.path.join(data_dir, self.train_file if filename is None else filename)))

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for hotpot versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("HotpotProcessor should be instantiated via HotpotV1Processor or HopotV2Processor")
        logging.info('dev file path is {}'.format(os.path.join(data_dir, self.dev_file if filename is None else filename)))
        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                sentence_indices = paragraph['sentence_indices']
                sentence_titles = paragraph['sentence_titles']
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []
                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if is_training:
                        answer = qa["answers"][0]
                        answer_text = answer.get('text', '')
                        start_position_character = answer.get('answer_start', -1)
                    answers = qa.get("answers", [])

                    example = HotpotExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        sentence_indices=sentence_indices,
                        sentence_titles=sentence_titles,
                    )

                    examples.append(example)
        return examples


class HotpotV1Processor(HotpotProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class HopotV2Processor(HotpotProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class HotpotExample(object):
    """
    A single training/test example for the hotpot dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        sentence_indices,
        sentence_titles,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers[:]
        self.sentence_indices=sentence_indices[:]
        self.sentence_titles=sentence_titles[:]

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class HotpotFeatures(object):
    """
    Single hotpot example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.hotpot.HotpotExample`
    using the :method:`~transformers.data.processors.hotpot.hotpot_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        span_sentence_indices,
        span_sentence_is_max,
        span_sentence_titles,
        sentence_tok_indices,
        type_label=None,
        level_label=None,
        yes_no_label=None,
        span_sentence_labels=None,
        span_sentence_indices_in_paragraph=None,
        sentence_token_indices=None,

    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position

        self.type_label = type_label
        self.yes_no_label = yes_no_label
        self.level_label = level_label
        self.span_sentence_labels = span_sentence_labels[:]
        self.span_sentence_is_max = span_sentence_is_max[:]
        self.span_sentence_titles = span_sentence_titles[:]
        self.span_sentence_indices = span_sentence_indices[:]
        self.sentence_tok_indices = sentence_tok_indices[:]
        self.span_sentence_indices_in_paragraph=span_sentence_indices_in_paragraph[:]
        self.sentence_token_indices = sentence_token_indices[:]


class HotpotResult(object):
    """
    Constructs a HotpotResult which can be used to evaluate a model's output on the hotpot dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, type_logits, sentence_logits, yes_no_logits,
                 start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        self.type_logit = type_logits
        self.sentence_logits = sentence_logits
        self.yes_no_logits = yes_no_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
