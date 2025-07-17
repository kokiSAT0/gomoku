"""自己対戦関数を小分けにしたサブモジュール集"""

from .policy_vs_policy import selfplay_policy_vs_policy
from .q_vs_q import selfplay_q_vs_q
from .policy_vs_q import selfplay_policy_vs_q

__all__ = [
    "selfplay_policy_vs_policy",
    "selfplay_q_vs_q",
    "selfplay_policy_vs_q",
]
