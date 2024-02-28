from __future__ import unicode_literals, print_function, division
import math, logging
import datasets
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from model import Model
from trainer import Trainer
from parse_args import parse_args

logger = logging.getLogger(__name__)

summarization_name_mapping = {
    #"ccdv/cnn_dailymail": ("article", "highlights"),
    "cnn_dailymail": ("article", "highlights"),
}

def prepare_dataset(args, tokenizer):
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(args.dataset_name,
                                name=args.dataset_config_name,
                                split='train',
                                #download_mode="force_redownload",
                                ignore_verifications=True,
                                cache_dir=args.dataset_cache_dir)
    logger.info("load_dataset succeeded.")
    # First we tokenize all the texts.
    # column_names = raw_datasets["train"].column_names
    column_names = raw_datasets.column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    prefix = args.source_prefix
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if args.n_train_data_samples == -1:
        sub_datasets = raw_datasets
    else:
        sub_datasets = raw_datasets.select(list(range(args.n_train_data_samples)))
    processed_datasets = sub_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_datasets #["train"]
    #train_dataset = processed_datasets["train"]
    return train_dataset

def main():
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()
    logger.info("All processes are synchronized.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
                                                  use_fast=not args.use_slow_tokenizer,
                                                  cache_dir=args.pretrained_model_cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    model_kwargs = {"vocab_size":len(tokenizer),
                    "h_dim":args.bilinear_dim,
                    "s_dim":args.bilinear_dim}
    model = Model(args, logger=logger, **model_kwargs)

    # https://huggingface.co/blog/pytorch-fsdp
    # prepare model before creating optimizer.
    model = accelerator.prepare(model)

    if model.seq2seq.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    train_dataset = prepare_dataset(args, tokenizer)

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Training
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Training
    train_processor = Trainer()
    train_processor(args=args,
                    train_dataloader=train_dataloader,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    accelerator=accelerator,
                    logger=logger)
    logger.info("Training is Done")


if __name__ == "__main__":
    main()
