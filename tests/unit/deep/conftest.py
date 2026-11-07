import pytest

pytest.importorskip('torch', reason='the deep families need the [torch] extra')
