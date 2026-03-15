import torch

from torchtitan.models.nanoVLM import model_registry
from torchtitan.models.nanoVLM.state_dict_adapter import NanoVLMStateDictAdapter


def test_soft_gating_momh_gate_exports_and_round_trips():
    model_config = model_registry("230m_momh_softgating").model
    adapter = NanoVLMStateDictAdapter(model_config, hf_assets_path=None)

    native_state_dict = {
        "layers.0.attn.momh_gate": torch.tensor(
            [[1.0, -1.0, 0.5, -0.5]],
            dtype=torch.float32,
        ),
    }

    hf_state_dict = adapter.to_hf(native_state_dict)

    assert "decoder.blocks.0.attn.momh_gate" in hf_state_dict
    assert torch.equal(
        hf_state_dict["decoder.blocks.0.attn.momh_gate"],
        native_state_dict["layers.0.attn.momh_gate"],
    )

    round_tripped = adapter.from_hf(hf_state_dict)

    assert "layers.0.attn.momh_gate" in round_tripped
    assert torch.equal(
        round_tripped["layers.0.attn.momh_gate"],
        native_state_dict["layers.0.attn.momh_gate"],
    )
