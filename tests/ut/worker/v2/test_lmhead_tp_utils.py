# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for draft LM-head TP alignment and sampling."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import lmhead_tp_utils
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import AscendGemma4Speculator
from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import AscendLiLiCorrSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator
from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator
from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import AscendStandaloneARSpeculator

MAX_REQS = 8
NUM_STEPS = 3
HIDDEN_SIZE = 4
VOCAB_SIZE = 5


def _speculator(cls=AscendMTPSpeculator):
    # Exercise production methods without loading weights or initializing NPUs.
    spec = cls.__new__(cls)
    spec.max_num_reqs = MAX_REQS
    spec.num_speculative_steps = NUM_STEPS
    spec.use_local_argmax_reduction = False
    spec.enable_adaptive_verification = False
    spec.use_fp64_gumbel = False
    spec.draft_watermarker = None
    spec.acceptance_estimator = None
    weight = torch.arange(HIDDEN_SIZE * VOCAB_SIZE, dtype=torch.float32).reshape(HIDDEN_SIZE, VOCAB_SIZE) / 10
    spec.model = SimpleNamespace(compute_logits=MagicMock(side_effect=lambda hidden: hidden @ weight))
    return spec, weight


@pytest.mark.parametrize(
    "speculator_cls,supported",
    [
        (AscendMTPSpeculator, True),
        (AscendEagleSpeculator, True),
        (AscendGemma4Speculator, True),
        (AscendDFlashSpeculator, True),
        (AscendDSparkSpeculator, True),
        # DFlash2's draft head call is collective-free and its candidate
        # top-k merge gathers over the lmhead-TP group on uniformly padded
        # draft-batch rows, so it is supported (head-level alignment).
        (AscendDFlash2Speculator, True),
        # LiLiCorr's draft collectives cannot be fed group-uniform rows yet.
        (AscendLiLiCorrSpeculator, False),
    ],
)
@pytest.mark.parametrize("draft_method", ["greedy", "probabilistic"])
def test_draft_sampling_support_scope(monkeypatch, speculator_cls, supported, draft_method):
    spec, _ = _speculator(speculator_cls)
    spec.speculative_config = SimpleNamespace(draft_sample_method=draft_method)
    monkeypatch.setattr(lmhead_tp_utils, "lmhead_tp_enable", lambda: True)
    if supported:
        spec._lmhead_tp_validate_draft_sampling()
    else:
        with pytest.raises(NotImplementedError, match=speculator_cls.__name__):
            spec._lmhead_tp_validate_draft_sampling()


@pytest.mark.parametrize("speculator_cls", [AscendStandaloneARSpeculator, AscendMultiModuleMTPSpeculator])
def test_unrelated_speculators_do_not_inherit_sampling_hook(speculator_cls):
    assert not issubclass(speculator_cls, lmhead_tp_utils.LmheadTPDraftSamplingMixin)
