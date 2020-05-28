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
""" Finetuning the library models for sequence classification on NLI Tasks (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import argparse
import glob
import json
import logging
import os
import random
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import balanced_accuracy_score

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    RobertaForTransferableMCQ,
    RobertaForTransferableEntailment,
    get_linear_schedule_with_warmup,
)

from utils_qa_as_nli import convert_examples_to_features
from utils_qa_as_nli import output_modes as output_modes
from utils_qa_as_nli import processors as processors, F1WithThreshold
from allennlp.training.metrics import F1Measure

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from wandb_utils import wandb, wandb_present, reset_output_dir, wandb_log  # noqa
from wandb_utils import init as wandb_init
logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (RobertaConfig, )),
    (),
)


class RobertaTokenizerRev(RobertaTokenizer):
    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           num_tokens_to_remove=0,
                           truncation_strategy="longest_first",
                           stride=0,
                           truncate_from_end=True):
        ids.reverse()

        if pair_ids is not None:
            pair_ids.reverse()
        ids, pair_ids, overflowing_tokens = super().truncate_sequences(
            ids,
            pair_ids,
            num_tokens_to_remove=num_tokens_to_remove,
            truncation_strategy=truncation_strategy,
            stride=stride)
        ids.reverse()

        if pair_ids is not None:
            pair_ids.reverse()
        overflowing_tokens.reverse()

        return (ids, pair_ids, overflowing_tokens)


MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForSequenceClassification,
                RobertaTokenizer),
    "roberta-mc": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    "roberta-rev": (RobertaConfig, RobertaForSequenceClassification,
                    RobertaTokenizerRev),
    "roberta-mc-rev": (RobertaConfig, RobertaForMultipleChoice,
                       RobertaTokenizerRev),
    "roberta-nli-transferable":
    (RobertaConfig, RobertaForTransferableEntailment, RobertaTokenizer),
    "roberta-nli-transferable-rev":
    (RobertaConfig, RobertaForTransferableEntailment, RobertaTokenizerRev),
    "roberta-mc-transferable": (RobertaConfig, RobertaForTransferableMCQ,
                                RobertaTokenizer),
    "roberta-mc-transferable-rev": (RobertaConfig, RobertaForTransferableMCQ,
                                    RobertaTokenizerRev),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def get_scores_using_threshold(score, threshold):
    score_for_positive_class = (score.detach().cpu() <= threshold).float()
    final_logits = torch.stack(
        (1.0 - score_for_positive_class, score_for_positive_class),
        dim=-1)  # (batch,2)

    return final_logits


def thresold_based_accuracy(scores, labels, threshold):
    preds = (scores <= threshold).astype('int')
    assert preds.dtype == labels.dtype

    return (preds == labels).mean(), preds


def balanced_accuracy(preds, labels):
    return balanced_accuracy_score(labels, preds)


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features]
            for feature in features]


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_name = args.task_name
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset = load_and_cache_examples(
        args,
        eval_task_name,
        tokenizer,
        evaluate=not args.test,
        test=args.test)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    #

    if args.use_threshold and args.threshold is None:
        # need to compute the threshold
        f1_metric = F1WithThreshold()
        logger.info("Using F1 with thresold computation")
        logger.info(
            "It will assume that the second entry in logits is the score")
    else:
        # args.use_threshold and args.threshold is not None
        # or
        # not args.use_threshold

        if not args.use_threshold:
            logger.info("Using regular F1 metric without threshold")
        else:  # not None
            logger.info(
                "Using regular F1 metrics with fixed threshold of {}".format(
                    args.threshold))
        f1_metric = F1Measure(positive_label=1)

    # Eval!
    logger.info("***** Running evaluation on {} set {} *****".format(
        "test" if args.test else "dev", prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    scores = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating", miniters=100):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2]

                    if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if scores is None:
            scores = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            scores = np.append(scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.use_threshold and args.threshold is None:
            # send neg class scores because threshold algorithm assumes
            # lesser the score better it is
            f1_metric(logits[:, 0].detach().cpu(),
                      inputs["labels"].detach().cpu())
        elif args.use_threshold and args.threshold is not None:
            # threshold is given. We use it to get preds first and use those as logits for eval
            final_logits = get_scores_using_threshold(logits[:, 0],
                                                      args.threshold)
            f1_metric(final_logits, inputs["labels"].detach().cpu())
        else:
            # not args.use_threshold => mcq or nli without threshold
            f1_metric(logits.detach().cpu(), inputs["labels"].detach().cpu())

    eval_loss = eval_loss / nb_eval_steps

    if not args.use_threshold:
        # mcq model or nli without threshold

        if args.output_mode == "classification":
            preds = np.argmax(scores, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(scores)

        acc = simple_accuracy(preds, out_label_ids)
        precision, recall, f1 = f1_metric.get_metric(reset=True)
        threshold = None
        balanced_acc = balanced_accuracy(preds, out_label_ids)
    elif args.threshold is not None:
        # using fixed threshold
        precision, recall, f1 = f1_metric.get_metric(reset=True)
        threshold = args.threshold
        # send neg class scores because threshold algorithm assumes
        # lesser the score better it is
        acc, preds = thresold_based_accuracy(scores[:, 0], out_label_ids,
                                             threshold)
        balanced_acc = balanced_accuracy(preds, out_label_ids)

    else:
        # using computed threshold
        precision, recall, f1, threshold = f1_metric.get_metric(reset=True)
        # send neg class scores because threshold algorithm assumes
        # lesser the score better it is
        acc, preds = thresold_based_accuracy(scores[:, 0], out_label_ids,
                                             threshold)
        balanced_acc = balanced_accuracy(preds, out_label_ids)

    if args.save_preds:
        pred_file = os.path.join(
            eval_output_dir, prefix,
            ("test" if args.test else "eval") + "_preds.txt")
        with open(pred_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(preds)
        score_file = os.path.join(
            eval_output_dir, prefix,
            ("test" if args.test else "eval") + "_scores.txt")
        with open(score_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(scores)
        label_file = os.path.join(
            eval_output_dir, prefix,
            ("test" if args.test else "eval") + "_labels.txt")
        with open(label_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(out_label_ids)

    p = "test" if args.test else "eval"
    result = {
        p + "_acc": acc,
        p + "_balanced_acc": balanced_acc,
        p + "_loss": eval_loss,
        p + "_f1": f1,
        p + "_precision": precision,
        p + "_recall": recall,
        p + "_threshold": threshold
    }

    results.update(result)

    output_eval_file = os.path.join(
        eval_output_dir, prefix,
        ("test" if args.test else "eval") + "_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** " + ("Test" if args.test else "Eval")
                    + " results {} *****".format(prefix))

        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]

    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_passage-{}_{}_{}_{}_{}_{}".format(
            ("static" if args.static_passage else "reg"),
            args.hypothesis_type,
            cached_mode,
            args.model_name_or_path.replace(
                '/', '_'
            ),  # have the full path to avoid mixing of checkpoints of different models
            str(args.max_seq_length),
            str(task),
        ),
    ) if not args.subset else os.path.join(
        args.data_dir,
        "cached_passage-{}_subset-{}_{}_{}_{}_{}_{}".format(
            ("static" if args.static_passage else "reg"),
            args.subset,
            args.hypothesis_type,
            cached_mode,
            args.model_name_or_path.replace(
                '/', '_'
            ),  # have the full path to avoid mixing of checkpoints of different models
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.num_choices)

        if evaluate:
            examples = processor.get_dev_examples(
                args.data_dir, args.hypothesis_type, args.subset,
                args.static_passage)
        elif test:
            examples = processor.get_test_examples(
                args.data_dir, args.hypothesis_type, args.subset,
                args.static_passage)
        else:
            examples = processor.get_train_examples(
                args.data_dir, args.hypothesis_type, args.subset,
                args.static_passage)

        features = convert_examples_to_features[task](
            examples,
            tokenizer,
            num_choices=args.num_choices,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(
                args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids(
                [tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            no_passage=args.no_passage,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset

    if task in ['single_choice']:
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                          dtype=torch.long)

        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features],
                                      dtype=torch.long)
    elif task in ['multiple_choice']:
        all_input_ids = torch.tensor(
            select_field(features, "input_ids"), dtype=torch.long)
        all_attention_mask = torch.tensor(
            select_field(features, "input_mask"), dtype=torch.long)
        all_token_type_ids = torch.tensor(
            select_field(features, "segment_ids"), dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features],
                                  dtype=torch.long)
    elif task in ['semantic_fragments']:
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([f.input_mask for f in features],
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([f.segment_ids for f in features],
                                          dtype=torch.long)
        all_labels = torch.tensor([f.label_id for f in features],
                                  dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_labels)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(
            MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--hypothesis_type",
        default=None,
        type=str,
        required=False,
        choices=['qa', 'rule', 'neural', 'hybrid'],
        help="The type of the hypothesis to use selected from the list: "
        + ", ".join(['qa', 'rule', 'neural', 'hybrid']),
    )
    parser.add_argument(
        "--num_choices",
        default=4,
        type=int,
        required=True,
        help="Number of answer options in the task.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        type=str,
        choices=['rule', 'neural'],
        help="Which subset of data to use.")
    parser.add_argument(
        '--test',
        action='store_true',
        help='evaluate on test set instead of dev')
    parser.add_argument(
        '--use_threshold',
        action='store_true',
        help='For nli models to use threshold instead of implicit default of 0.5')
    parser.add_argument(
        '--threshold',
        type=float,
        help="Use this threshold instead of finding one. Should always be given for --test and --use_threshold"
    )
    parser.add_argument(
        "--static_passage",
        action="store_true",
        help="Whether to use static passage.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions "
        "and checkpoints will be written. If using wandb, this will be ignored and output dir"
        " will be created by wandb and printed in the log.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action='store_true')

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        help="Whether to save predictions.")
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_passage",
        action="store_true",
        help="Set this flag if you training only using answer options. This can be used to validate model behavior and to check if options don't leak the answer."
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank")
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument(
        "--wandb", action="store_true", help="Use wandb or not")
    parser.add_argument(
        "--wandb_entity",
        default="ibm-cs696ds-2020",
        help="Team or username if using wandb")
    parser.add_argument(
        '--wandb_project',
        default="nli4qa",
        help="To set project if using non default project")
    parser.add_argument(
        '--wandb_runid',
        help="Run id. Unique to the project. "
        "Should be supplied if (and only if) resuming a wandb logged run."
        " For new runs, it is set by wandb.")

    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument(
        "--tags",
        default="",
        help="comma seperated (no space) list of tags for the run")

    args = parser.parse_args()

    # some validation

    if 'nli-transferable' in args.model_type:
        if not args.use_threshold:
            logger.warning(
                "Using NLI model but not using threshold."
                "\n This will not work if transfering from a mcq model")
        else:
            if args.test and (args.threshold is None):
                raise ValueError(
                    "threshold must be supplied if using threshold based test eval"
                )
    else:
        args.threshold = None  # force set it to None
        args.use_threshold = False
        logger.info(
            "Will not use threshold for non-binary classification. Will be ignored if supplied."
        )

    if args.wandb:
        args.tags = ','.join([args.task_name] + args.tags.split(","))
        wandb_init(args)
        args = reset_output_dir(args)

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))

    # Setup distant debugging if needed

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        False,
    )
    logger.info(f"Using  {args.n_gpu} gpus")

    # Set seed
    set_seed(args)

    # Prepare NLI task
    args.task_name = args.task_name.lower()

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.output_mode = output_modes[args.task_name]

    # Load pretrained model and tokenizer

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Evaluation
    results = {}

    if True:
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [args.model_name_or_path]

        for checkpoint in checkpoints:
            true_checkpoint = False

            if checkpoint.find("checkpoint") != -1:
                true_checkpoint = True

            global_step = checkpoint.split("-")[-1] if true_checkpoint else ""
            # if we find multiple checkpoints, means the output_dir is
            # parent of all ckpt dirs. We need the prefix then.
            prefix = checkpoint.split("/")[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result = evaluate(args, model, tokenizer, prefix=prefix)

            if global_step and args.wandb:
                step = None
                logger.debug(
                    f"Global step type={type(global_step)}, value= {global_step}"
                )
                try:
                    step = int(global_step)
                except ValueError as e:
                    logger.info("Global step not readable")
                    logger.error(e)
                    try:
                        step = json.loads('{"' + global_step)["step"]
                    except json.decoder.JSONDecodeError as je:
                        logger.error(je)
                        logger.error(
                            "Cannot read global step. Not logging to wandb")

                if step is not None:
                    wandb_log(result, step=step)
            result = dict(
                (k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
