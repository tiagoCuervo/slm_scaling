import pytest
import torch

from slm import ModelArgs, TransformerLM


def tiny_model():
    return TransformerLM(ModelArgs(dim=32, n_layers=2, n_heads=4, vocab_size=33, multiple_of=16, block_size=32))


def test_forward_backward_and_generate():
    model = tiny_model()
    x = torch.randint(0, 33, (2, 12))
    logits = model(x, x)
    assert logits.shape == (2, 12, 33)
    model.last_loss.backward()
    generated = model.eval().generate(x[:, :3], 4, temperature=0)
    assert generated.shape == (2, 7)


def test_cached_generation_matches_full_context():
    model = tiny_model().eval()
    prompt = torch.randint(0, 33, (2, 5))
    expected = prompt.clone()
    for _ in range(6):
        next_token = model(expected)[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, next_token), dim=1)
    actual = model.generate(prompt, 6, temperature=0)
    torch.testing.assert_close(actual, expected)


def test_cached_generation_with_grouped_query_attention():
    model = TransformerLM(
        ModelArgs(
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=33,
            multiple_of=16,
            block_size=32,
        )
    ).eval()
    prompt = torch.randint(0, 33, (2, 5))
    expected = prompt.clone()
    for _ in range(4):
        next_token = model(expected)[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, next_token), dim=1)
    torch.testing.assert_close(model.generate(prompt, 4, temperature=0), expected)


def test_cached_generation_with_padded_prefix():
    model = TransformerLM(
        ModelArgs(
            dim=32,
            n_layers=2,
            n_heads=4,
            vocab_size=33,
            prefix_vocab_size=8,
            prefix_pad=8,
            multiple_of=16,
            block_size=16,
        )
    ).eval()
    prompt = torch.tensor([[1, 2], [1, 2]])
    prefixes = torch.tensor([[4, 5, 8], [3, 4, 5]])
    expected = prompt.clone()
    for _ in range(3):
        next_token = model(expected, prefixes=prefixes)[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, next_token), dim=1)
    torch.testing.assert_close(
        model.generate(prompt, 3, prefixes=prefixes, temperature=0), expected
    )


def test_native_safetensors_roundtrip(tmp_path):
    model = tiny_model().eval()
    model.save_pretrained(tmp_path, max_shard_size=10_000)
    restored = TransformerLM.from_pretrained(tmp_path)
    x = torch.randint(0, 33, (2, 8))
    torch.testing.assert_close(model(x), restored(x))
    assert (tmp_path / "model.safetensors.index.json").exists()


def test_prefix_length_validation():
    model = TransformerLM(
        ModelArgs(
            dim=32,
            n_layers=1,
            n_heads=4,
            vocab_size=33,
            prefix_vocab_size=8,
            block_size=8,
            multiple_of=16,
        )
    )
    try:
        model.generate(torch.ones((1, 4), dtype=torch.long), 3, prefixes=torch.ones((1, 3), dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError("expected context-length validation")


def test_padded_prefix_matches_unpadded_prefix():
    model = TransformerLM(
        ModelArgs(
            dim=32,
            n_layers=1,
            n_heads=4,
            vocab_size=33,
            prefix_vocab_size=8,
            prefix_pad=8,
            block_size=12,
            multiple_of=16,
        )
    ).eval()
    tokens = torch.tensor([[1, 2, 3], [1, 2, 3]])
    padded = torch.tensor([[4, 5, 8], [3, 4, 5]])
    batched = model(tokens, prefixes=padded)
    single = model(tokens[:1], prefixes=torch.tensor([[4, 5]]))
    torch.testing.assert_close(batched[:1], single)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_new_tokens": -1}, "non-negative"),
        ({"max_new_tokens": 1, "temperature": float("nan")}, "finite"),
        ({"max_new_tokens": 1, "top_k": 0}, "positive"),
    ],
)
def test_generation_argument_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        tiny_model().generate(torch.ones((1, 2), dtype=torch.long), **kwargs)
