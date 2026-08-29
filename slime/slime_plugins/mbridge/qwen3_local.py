"""Allow the Qwen3 mbridge loader to use Megatron's local layer spec.

The upstream bridge maps Transformer Engine's fused layer-norm parameters.
The local spec stores the same two norms as standalone modules, so only their
parameter names differ.
"""

from mbridge.core import register_model
from mbridge.models.qwen3 import Qwen3Bridge as _Qwen3Bridge


@register_model("qwen3")
class Qwen3LocalBridge(_Qwen3Bridge):
    _OTHER_MAPPING = {
        "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
    }

    def _weight_name_mapping_mcore_to_hf(self, name):
        # ``pre_mlp_layernorm`` contains "mlp", so route it before the
        # generic bridge's MLP dispatch.
        if "pre_mlp_layernorm.weight" in name:
            layer = name.split(".")[2]
            return [f"model.layers.{layer}.post_attention_layernorm.weight"]
        return super()._weight_name_mapping_mcore_to_hf(name)
