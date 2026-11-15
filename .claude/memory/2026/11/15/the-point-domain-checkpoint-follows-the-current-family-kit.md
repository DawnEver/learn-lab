---
name: the-point-domain-checkpoint-follows-the-current-family-kit
description: Normal forge sync refreshes the stale tests-first branch kit and frozen environment; operator remains289 and actual API RED still belongs to GitHub Actions.
metadata:
  type: project
---

Root's normal pre-push refused the tests-only checkpoint be56937 because dev-lab lagged285 commits and lab-commons30, beyond20. This was a real PR-HEAD-IS-CURRENT refusal, not a transport/auth error. No hook was skipped and no push succeeded.

The named `python -m dev_lab.dev.forge sync` door ran in this independent OWN clone, starting clean with no live run. It refreshed the floated requirements and installed its own committed lock with `uv sync --frozen --extra torch --extra openvino`. The door automatically made normal hooked commit c9256d66e1d6f9069befe8400fbc47038c4b2159, including exactly the lock and its renderer output. No lock was handwritten.

| dependency | before | after |
|---|---|---|
| dev-lab | 6904a9ceb5781cd1eadf71d8c2f2d9e9046aff1c | bc55691a0575293fdc18355ae9f3cc4b4801e6c1 |
| lab-commons | de715ae4191c5aa0a7522198aae7153a6b91d850 | 2915f2d361a9867e29b866ab4a07dd7e1b86b284 |
| transitive numerics-lab | 04f6adac6068949dacabc4a02da68456e6f316b2 | 5aa9013b4535e9f142d4423869cfeaf23998e95a |

All-distributions environment key moved f7abb2e990f99e49 to62dcedff3a6b8259; the verdict key excluding the project moved0583ba7803d32f15 tocb3cff4e624102b4. Other repository environments were not changed. These new keys do not retroactively qualify any old measurement.

The normal renderer changed the guard engine/rules/settings, principles, ci.yml, merge-tick.yml and lock. Static `ciworkflows check`, `famfiles --files --check`, Ruff and format check all pass. The CI change grants actions:read to the absent-parts reader; merge-tick adds its declared record-pruning step. No workflow was hand-edited, dispatched or run by this worker.

The approved manual ci.yml dispatch remains safe for THIS non-integration branch with pr empty: HEAD_SHA is the branch SHA, full hosted parts run, security/head-gates/merge-ready require a PR and are off, the pending-integration step requires refs/heads/integrate/main and is off, and promote requires pending=true. The separate review-prepare workflow only handles pull_request_target. A branch push also triggers continue.yml, but the public continuation door creates a PR only when that same branch's previous PR merged; root must check the new branch has no such history before relying on no PR creation. No assertion is made that unrelated repository jobs cannot run independently.

All production src bytes still equal canonical289; operator.py blob is79078883982a7206269ac9840c733f476240a352. The tests-first checkpoint is still unfinished: no local API RED or pass, no operator implementation, no altered original training/bar, and no borrowed environment. Root owns normal publication and the existing GitHub Actions true-RED receipt before any production edit.
