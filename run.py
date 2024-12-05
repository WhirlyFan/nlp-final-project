import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Limit to save only the last checkpoint
    training_args.save_total_limit = 1

    # Dataset selection
    if args.dataset and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset else default_datasets[args.task]

        if dataset_id == ('glue', 'mnli'):
            eval_split = 'validation_matched'
        elif dataset_id == ('anli',):
            eval_split = 'dev'
        else:
            eval_split = 'validation'

        dataset = datasets.load_dataset(*dataset_id)

    # Initialize task-specific settings
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Load model and tokenizer
    model_classes = {'qa': AutoModelForQuestionAnswering, 'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Define preprocessing functions
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
    else:
        raise ValueError(f"Unrecognized task: {args.task}")

    # Preprocess datasets
    if dataset_id == ('snli',):
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
        train_dataset = dataset['train'] if training_args.do_train else None
        eval_dataset = dataset[eval_split] if training_args.do_eval else None
    elif dataset_id == ('anli',):
        train_dataset = datasets.concatenate_datasets([
            dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]
        ]) if training_args.do_train else None
        eval_dataset = datasets.concatenate_datasets([
            dataset["dev_r1"], dataset["dev_r2"], dataset["dev_r3"]
        ]) if training_args.do_eval else None
        if train_dataset:
            train_dataset = train_dataset.filter(lambda ex: ex['label'] != -1)
        if eval_dataset:
            eval_dataset = eval_dataset.filter(lambda ex: ex['label'] != -1)
    else:
        train_dataset = dataset['train'] if training_args.do_train else None
        eval_dataset = dataset[eval_split] if training_args.do_eval else None

    # Apply preprocessing
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Define trainer
    trainer_class = Trainer
    compute_metrics = compute_accuracy if args.task == 'nli' else None

    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate()
        print("Evaluation Results:", results)

        # Save evaluation metrics
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f)

        # Save predictions
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), 'w', encoding='utf-8') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example) + '\n')
            elif args.task == 'nli':
                for i, example in enumerate(eval_dataset):
                    example['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    main()
