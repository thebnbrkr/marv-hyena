import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiny_hyena import make_tiny  # noqa: E402

from marv_hyena import HyenaModel  # noqa: E402


@pytest.fixture(scope="module")
def hm():
    return HyenaModel(make_tiny())


@pytest.fixture
def seq():
    rng = random.Random(1)
    return "".join(rng.choice("ACGT") for _ in range(300))
