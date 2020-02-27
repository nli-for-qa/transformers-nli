from typing import Any, Dict
import logging
from pathlib import Path
import argparse
logger = logging.getLogger(__name__)
wandb = None
try:
    import wandb
    wandb_present = True
except ImportError:
    wandb_present = False

run = None


def init(args: Any) -> None:
    wandb_run = None
    fixed_ = dict(sync_tensorboard=True)
    tags = args.tags.split(',')

    if args.wandb and wandb_present:
        logger.info("Using wandb...")

        if args.wandb_project:
            if args.wandb_run_name:
                wandb_run = wandb.init(
                    name=args.wandb_run_name,
                    project=args.wandb_project,
                    tags=tags,
                    **fixed_)
            else:
                wandb_run = wandb.init(
                    project=args.wandb_project, tags=tags, **fixed_)
        else:
            if args.wandb_run_name:
                wandb_run = wandb.init(
                    name=args.wandb_run_name, tags=tags, **fixed_)
            else:
                wandb_run = wandb.init(tags=tags, **fixed_)
    wandb.config.update(args)

    return wandb_run


def reset_output_dir(args: argparse.Namespace) -> argparse.Namespace:
    if args.wandb and wandb_present:
        args.output_dir = str(
            (Path(wandb.run.dir) / 'training_dumps').absolute())
        logger.info(f"Reset the output dir to {args.output_dir}")

    return args


def wandb_log(metrics: Dict, step: int = None) -> None:
    if step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=step)
