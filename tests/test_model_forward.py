import torch
from torch import nn

from src.models.heads import MultiLabelClassificationHead
from src.models.pooling import MeanPooling
from src.models.registry import ModelOutput, SentenceClassificationModel


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16, seq_len: int = 6) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        hidden = torch.randn(batch_size, seq_len, self.hidden_size, device=input_ids.device)
        return type("Output", (), {"last_hidden_state": hidden})


def test_sentence_classification_forward_shapes():
    encoder = DummyEncoder()
    pooler = MeanPooling()
    classifier = MultiLabelClassificationHead(hidden_size=encoder.hidden_size, num_labels=3, dropout=0.0)
    model = SentenceClassificationModel(encoder=encoder, pooler=pooler, classifier=classifier)

    batch = {
        "input_ids": torch.randint(0, 5, (2, 4)),
        "attention_mask": torch.ones(2, 4),
    }
    output = model(**batch)
    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (2, 3)
    assert output.hidden_states.shape[0] == 2


def test_sentence_classification_autocast_cpu():
    encoder = DummyEncoder(hidden_size=8, seq_len=4)
    pooler = MeanPooling()
    classifier = MultiLabelClassificationHead(hidden_size=8, num_labels=2, dropout=0.0)
    model = SentenceClassificationModel(encoder=encoder, pooler=pooler, classifier=classifier)

    batch = {
        "input_ids": torch.randint(0, 3, (1, 3)),
        "attention_mask": torch.ones(1, 3),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(**batch)
    assert output.logits.shape == (1, 2)
    assert output.logits.dtype in {torch.float32, torch.bfloat16}
