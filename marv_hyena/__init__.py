"""marv-hyena: MARV-style interpretability for StripedHyena DNA models (Evo 2)."""
from .arch import BASES, KINDS, PARTS, HyenaModel, decode, encode
from .distance import DEFAULT_BANDS, DistanceResult, distance_profile, hyena_distance, show_profile
from .filters import Reach, lag_kernel, reach_table, reach_table_from_checkpoint, show_reach
from .intervene import (
    PatchSweep,
    kind_groups,
    mean_ablate,
    mean_writes,
    mixers_of,
    patch_sweep,
    span_accuracy,
    span_logprob,
    token_logprobs,
    zero_ablate,
)
from .codons import GENETIC_CODE, WobbleSite, translate, translation_test, wobble_sites
from .genome import RepeatFamily, find_repeat_families, repeat_probes, run_repeat_test
from .nullmodel import random_weights, verify_restored
from .probes import CopyProbe, Track, codon_phase_accuracy, copy_probe, genbank_track, score_copy, truncation_curve
from .trace import (
    Decomposition,
    Writes,
    capture_writes,
    decompose_prediction,
    decompose_writes,
    logit_direction,
    normed_direction,
)
from .variants import Variant, delta_logp, explain_variant, make_variant

__version__ = "0.1.0"
