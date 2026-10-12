"""Reinforcement learning package for the allocation engine.

Contents:

* ``features``  - the 13 pair features used by the policy (names are part of the
                  artifact contract);
* ``policy``    - the linear softmax policy plus artifact load/save;
* ``env``       - the allocation MDP (state, feasible actions, exact reward);
* ``train``     - REINFORCE training script (``python -m app.rl.train``);
* ``algorithm`` - the strategy wrapper that serves the trained policy.

The policy weights are a committed artifact: the container never trains, it only
loads ``artifacts/rl_policy.json``.
"""

from __future__ import annotations

__all__ = ["algorithm", "env", "features", "policy", "train"]
