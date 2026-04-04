import torch


class MultimodalCollator:
    """
    Universal collator for both CLIP and ViLT processors.

    Both CLIPProcessor and ViltProcessor expose the same
    (text, images) → {pixel_values, input_ids, attention_mask, ...} interface,
    so a single collator works for all model types.
    """

    def __init__(self, processor, max_length: int = 77) -> None:
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, batch):
        images, labels, texts = zip(*batch)

        inputs = self.processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs
