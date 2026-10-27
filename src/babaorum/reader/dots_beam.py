"""Faisceau correct pour dots.mocr sous transformers 4.51.3 (cause de la dégénérescence de la fumée P1 v2).

DotsOCRForCausalLM hérite de Qwen2ForCausalLM (modèle texte) et prend donc l'expansion GÉNÉRIQUE des entrées
pour le faisceau (GenerationMixin._expand_inputs_for_generation : repeat_interleave(k, dim=0) sur chaque
tenseur). Or pixel_values est la liste à plat des patchs de l'image [n_patchs, d] : chaque PATCH est répété k
fois de suite, et chacun des k faisceaux voit une image brouillée (le LLM écrit alors du chinois ou des lettres
sans ordre, comme dans la fumée). Qwen2.5-VL a sa propre méthode, qui répète l'image ENTIÈRE ; dots.mocr non.
patch_beam_expansion remplace la méthode du modèle : pour UNE image par échantillon et un lot de 1 (le cas des
carnets P1 : une ligne par generate()), pixel_values et image_grid_thw sont répétés par blocs entiers, le
reste par l'expansion générique.
"""

from __future__ import annotations

import types


def patch_beam_expansion(model) -> None:
    from transformers.generation.utils import GenerationMixin

    def _expand(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        if expand_size == 1:
            return input_ids, model_kwargs
        pv = model_kwargs.pop("pixel_values", None)
        grid = model_kwargs.pop("image_grid_thw", None)
        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            expand_size, is_encoder_decoder, input_ids, **model_kwargs)
        if pv is not None:
            assert grid is not None and grid.shape[0] == 1 and input_ids.shape[0] == expand_size, \
                "patch_beam_expansion : une image et un lot de 1 seulement"
            model_kwargs["pixel_values"] = pv.repeat(expand_size, *([1] * (pv.dim() - 1)))
            model_kwargs["image_grid_thw"] = grid.repeat(expand_size, 1)
        return input_ids, model_kwargs

    model._expand_inputs_for_generation = types.MethodType(_expand, model)
