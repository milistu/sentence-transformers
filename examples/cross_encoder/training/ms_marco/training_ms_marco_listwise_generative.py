import logging
from pathlib import Path

import torch
from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import ListwiseGenerativeLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)

LISTWISE_CHAT_TEMPLATE = """\
{% if messages[0].role == 'system' %}\
{{ '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}\
{% endif %}\
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Rank the documents by relevance to the query.") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
{% for message in messages %}{% if message.role.startswith("document_") %}\
<{{ message.role | replace("_", " ") | title }}>: {{ message.content }}
{% endif %}{% endfor %}<|im_end|>
<|im_start|>assistant
<think>

</think>

The most relevant document is:"""


def main() -> None:
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    cache_dir = Path(".cache")
    seed = 12

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    train_batch_size = 1
    eval_batch_size = 1
    num_epochs = 1
    max_docs = 3

    # 1. Define our CrossEncoder model
    torch.manual_seed(seed)
    model = CrossEncoder(
        model_name,
        reranking_mode="listwise",
        processor_kwargs={"chat_template": LISTWISE_CHAT_TEMPLATE},
        cache_folder=cache_dir,
    )
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    logging.info("Read train dataset")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train", cache_dir=cache_dir)

    def listwise_mapper(batch, max_docs: int | None = 10):
        processed_queries = []
        processed_docs = []
        processed_labels = []

        for query, passages_info in zip(batch["query"], batch["passages"]):
            passages = passages_info["passage_text"]
            labels = passages_info["is_selected"]

            paired = sorted(zip(passages, labels), key=lambda x: x[1], reverse=True)
            sorted_passages, sorted_labels = zip(*paired) if paired else ([], [])

            if max(sorted_labels) < 1.0:
                continue

            if max_docs is not None:
                sorted_passages = list(sorted_passages[:max_docs])
                sorted_labels = list(sorted_labels[:max_docs])

            processed_queries.append(query)
            processed_docs.append(sorted_passages)
            processed_labels.append(sorted_labels)

        return {
            "query": processed_queries,
            "docs": processed_docs,
            "labels": processed_labels,
        }

    dataset = dataset.map(
        lambda batch: listwise_mapper(batch=batch, max_docs=max_docs),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing listwise samples",
    )

    dataset = dataset.shuffle(seed=seed).select(range(1000))
    dataset = dataset.train_test_split(test_size=100, seed=seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = ListwiseGenerativeLoss(model)

    # 4. Define the evaluator
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco"],
        batch_size=eval_batch_size,
    )
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name.split("/")[-1]
    run_name = f"reranker-msmarco-v1.1-{short_model_name}-listwise-generative"
    args = CrossEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_steps=0.1,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        load_best_model_at_end=False,
        eval_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        logging_steps=1,
        logging_first_step=True,
        max_steps=50,
        run_name=run_name,
        seed=seed,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        evaluator=evaluator,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    # 7. Evaluate the final model
    # evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
