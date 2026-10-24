"""HF configuration checks without Torch, model downloads or accelerators."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def hf(monkeypatch):
    class Device(str):
        @property
        def type(self):
            return self.split(":")[0]

    torch = SimpleNamespace(
        device=Device, float32="float32", float16="float16", bfloat16="bfloat16",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    transformers = MagicMock()
    model = transformers.AutoModelForImageTextToText.from_pretrained.return_value
    model.to.return_value = model
    model.eval.return_value = model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    path = Path(__file__).resolve().parents[1] / "providers/hf.py"
    spec = importlib.util.spec_from_file_location("providers.hf", path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    monkeypatch.setitem(sys.modules, "providers.hf", provider)
    return provider, transformers


def test_hf_cache_distinguishes_device_and_precision(hf):
    provider, transformers = hf
    first = provider.load_model("test-vlm", device="cpu", dtype="float32")
    assert provider.load_model("test-vlm", device="auto", dtype="auto") is first
    provider.load_model("test-vlm", device="cuda:0", dtype="float32")
    provider.load_model("test-vlm", device="cpu", dtype="bfloat16")
    loads = transformers.AutoModelForImageTextToText.from_pretrained.call_args_list
    assert [call.kwargs["dtype"] for call in loads] == ["float32", "float32", "bfloat16"]
    assert [str(key[1]) for key in provider._LOADED_MODELS] == ["cpu", "cuda:0", "cpu"]
    with pytest.raises(ValueError, match="Unknown HF dtype"):
        provider.load_model("test-vlm", dtype="typo")
    assert len(loads) == 3


@pytest.mark.parametrize("device", ["mps", "cuda", "cpu"])
def test_hf_auto_preserves_device_priority_and_precision(hf, monkeypatch, device):
    provider, transformers = hf
    monkeypatch.setattr(provider.torch.backends.mps, "is_available", lambda: device == "mps")
    monkeypatch.setattr(provider.torch.cuda, "is_available", lambda: device != "cpu")
    _, _, resolved = provider.load_model("test-vlm", device="auto", dtype="auto")
    assert resolved.type == device
    expected = "float32" if device == "cpu" else "bfloat16"
    assert transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs["dtype"] == expected


def test_hf_navigation_forwards_settings_to_model_loader(hf, monkeypatch):
    from llm.navigation.service import get_next_move

    provider, _ = hf
    # Stop at loading: exercise navigation, dispatch and HF chat without inference.
    loader = MagicMock(side_effect=RuntimeError("loading reached"))
    monkeypatch.setattr(provider, "load_model", loader)
    with pytest.raises(RuntimeError, match="loading reached"):
        get_next_move(
            "Choose a cube.", model="test-vlm", client_type="hf",
            image_path="observation.jpg", hf_device="cuda:1", hf_dtype="float16",
        )
    loader.assert_called_once_with("test-vlm", device="cuda:1", dtype="float16")


def test_main_passes_hf_configuration_through_runtime_to_loader(hf, monkeypatch, tmp_path):
    import main
    from hydra import compose, initialize_config_dir
    from runtime.controller import Controller

    with initialize_config_dir(config_dir=str(Path(main.__file__).parent / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "experiment=e1", "client=hf", "nav=test-vlm", "hf.device=cpu",
            "hf.dtype=float32", "hf.max_output_tokens=17", f"outputs.dir={tmp_path}",
        ])
    env = MagicMock(position=(0.0, 0.0), direction="N", object_positions_2d={})
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: env)
    env.capture_image.side_effect = lambda path: path
    provider, _ = hf
    loader = MagicMock(side_effect=RuntimeError("loading reached"))
    monkeypatch.setattr(provider, "load_model", loader)
    checked = []

    def check_runtime(controller):
        checked.append((controller.hf_device, controller.hf_dtype, controller.provider_max_output_tokens))
        controller.current_obs_dir = tmp_path
        controller._new_outcome = lambda *args: None
        controller._finalize_episode = lambda *args: None
        controller._run_single_episode(1)
        loader.assert_called_once_with("test-vlm", device="cpu", dtype="float32")
        controller._close_resources()

    monkeypatch.setattr(Controller, "run", check_runtime)
    main.main.__wrapped__(cfg)
    assert checked == [("cpu", "float32", 17)]
