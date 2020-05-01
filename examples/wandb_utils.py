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
    """Current init signature in wandb client:
         init(job_type=None, dir=None, config=None, project=None, entity=None, reinit=None, tags=None,
         group=None, allow_val_change=False, resume=False, force=False, tensorboard=False,
         sync_tensorboard=False, monitor_gym=False, name=None, notes=None, id=None, magic=None,
         anonymous=None, config_exclude_keys=None, config_include_keys=None):

        See: https://github.com/wandb/client/blob/e3e4ef24b5c246f1d54db24427260b0e6cd2aa6d/wandb/__init__.py
    """
    wandb_run = None
    fixed_ = dict(sync_tensorboard=True)

    if args.tags:
        tags = args.tags.split(',')
    else:
        tags = None

    if args.wandb and wandb_present:
        logger.info("Using wandb...")

        if args.resume and (args.wandb_runid is None):
            raise ValueError(
                "--wandb_runid has to be supplied if --wandb and --resume are true"
            )

        if args.resume and args.wandb_run_name:
            logger.info(
                f"Ignoring provided run name {args.wandb_run_name} because --resume was set"
            )
            args.wandb_run_name = None
        wandb_run = wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_runid,
            tags=tags,
            resume="must"

            if args.resume else False,  # "must" not "True", retains run name
            **fixed_)

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

# def _check_files(p: Path, filelist: str) -> List:
# for f in filelist:
# (p / f).isfile()  # dhruvesh

# def wet_directory(path: str, args: argparse.Namespace):
    """Check if the directory has the following. If it does not
    then we try brining the same from wandb if resuming (not doing this
    for now)

        If doing transfer learning:

            config.json (model config)
            pytorch_model.bin (model weights)
            tokenizer_config.json
            vocab.json
            special_tokens_map.json
            merges.txt

        If resuming:
            all the ones above plus
            scheduler.pt
            optimizer.pt
    """


#    path = Path(path)

#    for
