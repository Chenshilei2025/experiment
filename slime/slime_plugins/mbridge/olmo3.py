"""Megatron/HF bridge for OLMo-3 dense models."""

from mbridge.core import register_model
from mbridge.models import Qwen2Bridge


@register_model("olmo3")
class OLMo3Bridge(Qwen2Bridge):
    _OTHER_MAPPING = {
        "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
        "post_attention_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "post_feedforward_layernorm.weight": ["model.layers.{layer_number}.post_mlp_layernorm.weight"],
    }

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            qk_layernorm=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            post_mlp_layernorm=True,
            post_self_attn_layernorm=True,
            rotary_interleaved=False,
        )

