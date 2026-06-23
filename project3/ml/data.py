"""
AG News data loading.

Two sources are supported:
  1. HuggingFace `datasets` (the official route from the project brief):
         load_dataset("fancyzhx/ag_news")
     Use load_agnews(source="hf").
  2. Local CSV files in the classic AG-News format
         label,title,description     (label in {1,2,3,4})
     Use load_agnews(source="csv", data_dir=...).

In both cases we return a simple, framework-agnostic container of
python lists, so the rest of the code never depends on which source
was used.

Labels are normalised to 0..3 internally with the names:
    0: World, 1: Sports, 2: Business, 3: Sci/Tech
"""

from dataclasses import dataclass
import csv
import os

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class TextDataset:
    texts: list      # list[str]
    labels: list     # list[int] in 0..3

    def __len__(self):
        return len(self.texts)


def _read_classic_csv(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label = int(row[0]) - 1            # 1..4 -> 0..3
            title, desc = row[1], row[2]
            text = (title + " " + desc).replace("\\", " ").strip()
            texts.append(text)
            labels.append(label)
    return TextDataset(texts, labels)


def load_agnews(source="hf", data_dir=None):
    """Return (train_ds, test_ds) as TextDataset objects."""
    if source == "csv":
        if data_dir is None:
            raise ValueError("data_dir required for source='csv'")
        train = _read_classic_csv(os.path.join(data_dir, "train.csv"))
        test = _read_classic_csv(os.path.join(data_dir, "test.csv"))
        return train, test

    if source == "hf":
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")

        def conv(split):
            return TextDataset(list(split["text"]), list(split["label"]))

        return conv(ds["train"]), conv(ds["test"])

    raise ValueError(f"Unknown source: {source}")


def subset(ds, idx):
    """Return a TextDataset restricted to the given indices."""
    return TextDataset([ds.texts[i] for i in idx],
                       [ds.labels[i] for i in idx])
