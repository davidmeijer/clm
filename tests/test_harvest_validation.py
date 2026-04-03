import json

import torch

import harvest.validation as validation


class DummyModel:
    def __init__(self):
        self.sample_batch_sizes = []

    def eval(self):
        return self

    def sample(self, *, descriptors=None, n_sequences=None, max_len=250):
        assert not torch.is_grad_enabled()
        self.sample_batch_sizes.append(n_sequences)
        return ["CCO"] * n_sequences


class DummyModelConfig:
    def __init__(self, condition_vocab_path, test_path, model):
        self.condition_vocab_path = condition_vocab_path
        self.test_path = test_path
        self._model = model

    def load_model(self, device):
        return self._model


def test_validate_uses_inference_mode_and_respects_batch_size(
    monkeypatch, tmp_path
):
    condition_vocab_path = tmp_path / "condition_vocab.json"
    condition_vocab_path.write_text(json.dumps({"labels": ["<UNK>", "Ala"]}))

    test_path = tmp_path / "test.jsonl"
    test_row = {
        "smiles": "CCO",
        "condition_graph": {
            "nodes": [{"id": 0, "name": "Ala"}],
            "links": [],
        },
    }
    test_path.write_text(json.dumps(test_row) + "\n")

    model = DummyModel()
    model_cfg = DummyModelConfig(condition_vocab_path, test_path, model)
    monkeypatch.setattr(validation, "prep_clm", lambda model_dir, eval=True: [model_cfg])

    validation.cmd_validate(
        model_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        device="cpu",
        test_size=1,
        sample_size=5,
        batch_size=2,
    )

    assert model.sample_batch_sizes == [2, 2, 1]
    result_file = tmp_path / "out" / "validation_results.tsv"
    assert result_file.is_file()
    assert len(result_file.read_text().strip().splitlines()) == 2
