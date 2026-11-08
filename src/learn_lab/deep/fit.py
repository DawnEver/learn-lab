"""The one training loop: Adam, then optional L-BFGS polish, on a loss closure; the history is the record.

A :class:`Checkpoint` makes the Adam phase resumable. Every ``every`` steps it writes, into its directory,
``weights.pt`` -- the parameters, the Adam state and the torch RNG state, saved by ``torch.save`` and read
back ONLY through ``torch.load(weights_only=True)`` (tensors and plain containers, never an arbitrary
object) -- and ``checkpoint.json``: the step, the loss history, the config identity, the payload's SHA-256 and
the record's own content hash (the scheme of :mod:`learn_lab.artifact`). A fit given a directory that holds a
checkpoint resumes from it after verifying both hashes and that it was written by the SAME config (the
caller's identity, the schedule and the parameter shapes); anything else is refused. Under
:meth:`Runtime.seeded`'s deterministic kernels the resumed fit ends on the uninterrupted fit's weights bit for
bit. The L-BFGS polish is one optimizer call and is not checkpointed: a fit interrupted inside it resumes
from the end of the Adam phase.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from learn_lab.errors import refuse

__all__ = ['Checkpoint', 'Schedule', 'minimise']

FORMAT = 'learn-lab fit checkpoint'
VERSION = 1
MANIFEST = 'checkpoint.json'
WEIGHTS = 'weights.pt'


@dataclass(frozen=True, slots=True)
class Schedule:
    """``adam_steps`` at ``learning_rate``, then ``lbfgs_steps`` L-BFGS iterations (0 skips the polish)."""

    adam_steps: int = 2000
    learning_rate: float = 1e-3
    lbfgs_steps: int = 0

    def __post_init__(self) -> None:
        if self.adam_steps < 0 or self.lbfgs_steps < 0 or self.adam_steps + self.lbfgs_steps == 0:
            refuse(f'a schedule takes at least one step, got adam={self.adam_steps} lbfgs={self.lbfgs_steps}')
        if not self.learning_rate > 0:
            refuse(f'a learning rate is > 0, got {self.learning_rate}')


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Write the fit's state to *directory* every *every* Adam steps, and resume from it when it holds one."""

    directory: Path
    every: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'directory', Path(self.directory))
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            refuse(f'a checkpoint is written every n >= 1 steps, got {self.every!r}')

    def member(self, index: int) -> Checkpoint:
        """The checkpoint of member *index* of a fit that trains several networks (a subdirectory)."""
        return replace(self, directory=self.directory / f'member-{index}')


def _sha256(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()


def _identity(config: str, schedule: Schedule, parameters: list[torch.Tensor]) -> str:
    shapes = [[str(p.dtype), list(p.shape)] for p in parameters]
    return _sha256(_canonical({'config': config, 'schedule': repr(schedule), 'parameters': shapes}))


def _rng() -> dict[str, object]:
    state: dict[str, object] = {'cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _write(checkpoint: Checkpoint, identity: str, parameters, adam, history: list[float]) -> None:
    directory = checkpoint.directory
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        'parameters': [p.detach().to('cpu', copy=True) for p in parameters],
        'optimizer': adam.state_dict(),
        'rng': _rng(),
    }
    torch.save(state, directory / f'{WEIGHTS}.tmp')
    (directory / f'{WEIGHTS}.tmp').replace(directory / WEIGHTS)
    record = {
        'format': FORMAT,
        'version': VERSION,
        'config': identity,
        'step': len(history),
        'history': history,
        'payload': {WEIGHTS: _sha256((directory / WEIGHTS).read_bytes())},
    }
    record['content_hash'] = _sha256(_canonical(record))
    (directory / f'{MANIFEST}.tmp').write_text(json.dumps(record, indent=1, sort_keys=True), encoding='utf-8')
    (directory / f'{MANIFEST}.tmp').replace(directory / MANIFEST)


def _read(checkpoint: Checkpoint, identity: str) -> tuple[dict, list[float]] | None:
    """The verified state and history in *checkpoint*'s directory; ``None`` when it holds no checkpoint."""
    path = checkpoint.directory / MANIFEST
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        refuse(f'{path} is not a readable checkpoint manifest: {exc}')
    if not isinstance(record, dict) or record.get('format') != FORMAT or record.get('version') != VERSION:
        refuse(f'{path} is not a {FORMAT} version {VERSION}')
    claimed = record.pop('content_hash', None)
    if claimed != _sha256(_canonical(record)):
        refuse(f'{path} does not match its content hash: the manifest was changed after it was written')
    if record['config'] != identity:
        refuse(f'{checkpoint.directory} holds a checkpoint of another config; resume only the fit that wrote it')
    weights = checkpoint.directory / WEIGHTS
    if not weights.exists() or _sha256(weights.read_bytes()) != record['payload'][WEIGHTS]:
        refuse(f'{weights} does not match its hash in the manifest: the payload was changed after it was written')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    history = record['history']
    if len(history) != record['step']:
        refuse(f'{path} records step {record["step"]} with {len(history)} losses')
    return state, history


def minimise(
    parameters: Iterable[torch.Tensor],
    loss: Callable[[], torch.Tensor],
    schedule: Schedule,
    *,
    checkpoint: Checkpoint | None = None,
    config: str = '',
) -> list[float]:
    """Minimise ``loss()`` over *parameters*; return the loss after every step. A non-finite loss is refused.

    With a *checkpoint*, the Adam phase writes its state every ``checkpoint.every`` steps (and at its end) and
    resumes from a checkpoint already in the directory. *config* is the caller's identity of what is being
    fitted (a family passes its declaration's repr); a checkpoint written under another *config*, schedule or
    parameter shapes is refused.
    """
    parameters = list(parameters)
    history: list[float] = []

    def record(value: torch.Tensor) -> None:
        number = float(value.detach())
        if not math.isfinite(number):
            refuse(f'the training loss became {number} at step {len(history)}')
        history.append(number)

    adam = torch.optim.Adam(parameters, lr=schedule.learning_rate)
    identity = '' if checkpoint is None else _identity(config, schedule, parameters)
    resumed = None if checkpoint is None else _read(checkpoint, identity)
    if resumed is not None:
        state, history = resumed[0], list(resumed[1])
        with torch.no_grad():
            for p, saved in zip(parameters, state['parameters'], strict=True):
                p.copy_(saved)
        adam.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']['cpu'])
        if 'cuda' in state['rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    for _ in range(len(history), schedule.adam_steps):
        adam.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        adam.step()
        record(value)
        if checkpoint is not None and (len(history) % checkpoint.every == 0 or len(history) == schedule.adam_steps):
            _write(checkpoint, identity, parameters, adam, history)
    if schedule.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            parameters,
            max_iter=schedule.lbfgs_steps,
            line_search_fn='strong_wolfe',
            tolerance_grad=0.0,
            tolerance_change=0.0,
            history_size=50,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            value = loss()
            value.backward()
            return value

        lbfgs.step(closure)
        # Not under no_grad: a physics loss differentiates the field wrt its inputs to form the residual.
        record(loss())
    return history
