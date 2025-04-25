import evaluate
from datasets import load_dataset, Features, ClassLabel, Image, DownloadConfig, Dataset
from fiftyone.types import BDDDataset
from fiftyone.utils.bdd import BDDDatasetImporter
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, \
    default_data_collator
import fiftyone.utils.huggingface as fouh
import fiftyone as fo


def main():
    FO_NAME = "bdd100k_weather"
    if FO_NAME in fo.list_datasets():
        fo.delete_dataset(FO_NAME)
    ds = fouh.load_from_hub("dgural/bdd100k")
    filepaths = ds.values("filepath")
    weather_labels = ds.values("weather.label")
    ds = Dataset.from_dict({"pixel_values": filepaths, "label": weather_labels})
    classes = sorted(set(weather_labels))
    new_feature = Features({
        "pixel_values": Image(),
        "label": ClassLabel(names=classes)
    })
    ds = ds.cast(new_feature)
    to_remove = "undefined"
    class_idx = ds.features["label"].str2int(to_remove)
    ds = ds.filter(lambda x: x["label"] != class_idx)
    ds = ds.train_test_split(test_size=0.2)
    train_ds, val_ds = ds['train'], ds['test']

    def preprocess(sample):
        inputs = processor(sample["pixel_values"], return_tensors="pt")
        sample["pixel_values"] = inputs["pixel_values"]
        sample["label"] = sample["label"]
        return sample

    train_ds = train_ds.with_transform(preprocess)
    val_ds = val_ds.with_transform(preprocess)

    model_name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(train_ds.features["label"].names),
        id2label={i: lbl for i, lbl in enumerate(train_ds.features["label"].names)},
        label2id={lbl: i for i, lbl in enumerate(train_ds.features["label"].names)},
        ignore_mismatched_sizes=True,
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./weather_classifier",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./weather_classifier")

if __name__ == "__main__":
    main()