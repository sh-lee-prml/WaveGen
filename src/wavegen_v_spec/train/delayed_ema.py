"""Delayed initialization, otherwise the original ema-pytorch averaging."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from ema_pytorch import EMA


EMA_START_UPDATE = 2000
EMA_DECAY = 0.9999
EMA_UPDATE_EVERY = 10
EMA_CONTRACT = {
    "implementation": "ema_pytorch.EMA_original_SF_defaults",
    "start_after_completed_G_update": EMA_START_UPDATE,
    "initialization": "exact_Online_copy_after_update2000_no_source_EMA_history",
    "decay": EMA_DECAY,
    "update_every": EMA_UPDATE_EVERY,
    "update_after_step": 100,
    "inv_gamma": 1.0,
    "power": 2 / 3,
    "min_value": 0.0,
    "decay_warmup": True,
    "counter": "completed_G_updates_not_D_updates",
    "serialized_step": "native_update_calls_G_stage_minus2000_plus1_zero_before_initialization",
    "age": "completed_G_stage_update_minus2000_zero_before_initialization",
    "owner": "global_main_rank_only",
    "parameters_dtype": "torch.float32",
    "frozen_parameters": "native_EMA_all_floating_parameters_including_frozen",
    "buffers": "native_EMA_floating_buffer_copy_or_average_on_10_update_cadence",
    "nonfloating_buffers": "initial_snapshot_as_in_original_SF_EMA",
    "parameter_aliases": "native_named_parameters_deduplicated",
    "discriminator": False,
}


def _update(value):
    if type(value) is not int or value < 0:
        raise ValueError("EMA G update must be a nonnegative integer")
    return value


class DelayedGeneratorEMA:
    """Call after every successful G update; EMA itself handles its cadence.

    The first call to native ``EMA.update`` is after G update2000. It copies
    Online exactly and leaves native ``step=1`` (averaging ``age=0``). The
    original SF EMA then copies every ten steps during its100-step warmup,
    and uses its unchanged inverse-power decay schedule capped at0.9999.
    Non-main ranks track only counters, never allocate a second generator.
    """

    def __init__(self, model, *, enabled: bool):
        if type(enabled) is not bool:
            raise TypeError("EMA rank ownership must be an explicit boolean")
        self.enabled = enabled
        self.last_g_stage_update = 0
        self._model = model
        self._ema = None
        state = model.state_dict(keep_vars=True)
        if not state:
            raise ValueError("EMA needs the full nonempty generator state")
        for parameter in model.parameters():
            if parameter.dtype != torch.float32:
                raise ValueError("EMA requires FP32 generator master parameters")
        first, self._aliases, self._schema = {}, {}, {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                raise TypeError("EMA supports tensor generator state only")
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError("EMA requires FP32 floating generator buffers")
            self._aliases[key] = first.setdefault(id(value), key)
            self._schema[key] = (value.shape, value.dtype)

    @staticmethod
    def _new_native(model):
        # These are the original SF EMA defaults, not a fixed-decay rewrite.
        return EMA(model, include_online_model=False, beta=EMA_DECAY,
                   update_every=EMA_UPDATE_EVERY, update_after_step=100,
                   inv_gamma=1.0, power=2 / 3, min_value=0.0,
                   lazy_init_ema=True)

    @property
    def initialized(self):
        return self.last_g_stage_update >= EMA_START_UPDATE

    @property
    def age(self):
        return max(0, self.last_g_stage_update - EMA_START_UPDATE)

    @property
    def native_step(self):
        return self.age + 1 if self.initialized else 0

    @property
    def shadow(self):
        return None if self._ema is None else self._ema.ema_model.state_dict()

    def tracking_state(self):
        return {"contract": dict(EMA_CONTRACT), "initialized": self.initialized,
                "age": self.age, "last_g_stage_update": self.last_g_stage_update,
                "native_step": self.native_step}

    @torch.no_grad()
    def after_step(self, stage_update: int):
        """Call once, only AFTER a successful committed generator update."""
        _update(stage_update)
        if stage_update != self.last_g_stage_update + 1:
            raise RuntimeError("EMA must observe every successful G update exactly once")
        if self.enabled and stage_update >= EMA_START_UPDATE:
            if stage_update == EMA_START_UPDATE:
                self._ema = self._new_native(self._model)
            elif self._ema is None:
                raise RuntimeError("Missing initialized EMA snapshot")
            self._ema.update()
        self.last_g_stage_update = stage_update

    def state_dict(self):
        if not self.enabled:
            raise RuntimeError("Only the global main rank may export generator EMA")
        if not self.initialized:
            if self._ema is not None:
                raise RuntimeError("EMA shadow must not exist before update2000")
            return {"initted": torch.tensor(False, dtype=torch.bool),
                    "step": torch.tensor(0, dtype=torch.int64)}
        if (self._ema is None or not self._ema.initted.item()
                or self._ema.step.item() != self.native_step):
            raise RuntimeError("Native EMA counter does not match committed generator updates")
        return self._ema.state_dict()

    @torch.no_grad()
    def load_state_dict(self, state, *, stage_update: int, tracking):
        """Resume this exact native-EMA contract; never reset old EMA history."""
        _update(stage_update)
        initialized = stage_update >= EMA_START_UPDATE
        age = max(0, stage_update - EMA_START_UPDATE)
        native_step = age + 1 if initialized else 0
        expected_tracking = {"contract": dict(EMA_CONTRACT), "initialized": initialized,
                             "age": age, "last_g_stage_update": stage_update,
                             "native_step": native_step}
        if (not isinstance(tracking, Mapping) or tracking != expected_tracking
                or type(tracking.get("initialized")) is not bool
                or any(type(tracking.get(key)) is not int
                       for key in ("age", "last_g_stage_update", "native_step"))):
            raise RuntimeError("Delayed EMA tracking counter/contract differs; old EMA history cannot be reconstructed")
        keys = {"step", "initted"}
        if initialized:
            keys |= {"ema_model." + key for key in self._schema}
        if not isinstance(state, Mapping) or set(state) != keys:
            raise RuntimeError("Delayed EMA full-generator keys differ from initialization state")
        for key, dtype, expected in (("step", torch.int64, native_step),
                                     ("initted", torch.bool, initialized)):
            tensor = state[key]
            if (not torch.is_tensor(tensor) or tensor.ndim != 0 or tensor.dtype != dtype
                    or tensor.item() != expected):
                raise RuntimeError("Delayed EMA native step is its update-call count, not age or absolute update")
        if initialized:
            for key, (shape, dtype) in self._schema.items():
                value = state["ema_model." + key]
                if not torch.is_tensor(value) or value.shape != shape or value.dtype != dtype:
                    raise RuntimeError(f"Delayed EMA shape/dtype differs: {key}")
                canonical = self._aliases[key]
                if canonical != key and not torch.equal(value, state["ema_model." + canonical]):
                    raise RuntimeError(f"Delayed EMA duplicated generator alias differs: {key}")
        native = None
        if self.enabled and initialized:
            native = self._new_native(self._model)
            native.init_ema()
            native.load_state_dict(state, strict=True)
        self._ema = native
        self.last_g_stage_update = stage_update
