"""エージェント生成を司るファクトリ関数を提供するモジュール"""

from pathlib import Path

from ..ai.agents import (
    PolicyAgent,
    QAgent,
    RandomAgent,
    ImmediateWinBlockAgent,
    FourThreePriorityAgent,
    LongestChainAgent,
)


def create_agent(agent_type: str, board_size: int, agent_path: Path | None = None, agent_params: dict | None = None):
    """
    agent_type の文字列から適切なエージェントインスタンスを生成する。

    - ``policy`` / ``q`` などモデル読み込みを伴うタイプは ``agent_path`` を用いる。
    - ``agent_params`` に ``network_type`` を与えるとネットワーク形式を切り替えられる。
    - 指定が不明な場合は ``RandomAgent`` を返す。
    """
    if agent_params is None:
        agent_params = {}

    key = agent_type.lower()
    alias = {
        "rand": "random",
        "immediatewinblockagent": "immediate",
        "fourthreepriorityagent": "fourthree",
        "longestchainagent": "longest",
    }
    key = alias.get(key, key)

    class_table = {
        "policy": PolicyAgent,
        "q": QAgent,
        "random": RandomAgent,
        "immediate": ImmediateWinBlockAgent,
        "fourthree": FourThreePriorityAgent,
        "longest": LongestChainAgent,
    }

    if key in ("policy", "q"):
        # 学習済みモデルを読み込むタイプ
        agent = class_table[key](board_size=board_size, **agent_params)
        if agent_path:
            agent.load_model(agent_path)
        return agent

    if key in class_table:
        # ヒューリスティック系やランダムエージェント
        return class_table[key]()

    # 不明な文字列の場合は警告を出してランダムにフォールバック
    print(f"[WARNING] Unknown agent_type={agent_type}, fallback to RandomAgent.")
    return RandomAgent()

