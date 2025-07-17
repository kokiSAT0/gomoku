# -*- coding: utf-8 -*-
"""自己対戦関連の関数を分割した新モジュールへのラッパー"""

from .selfplay import (
    selfplay_policy_vs_policy,
    selfplay_q_vs_q,
    selfplay_policy_vs_q,
)

__all__ = [
    "selfplay_policy_vs_policy",
    "selfplay_q_vs_q",
    "selfplay_policy_vs_q",
]
