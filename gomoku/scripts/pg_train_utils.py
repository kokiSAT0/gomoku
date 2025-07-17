"""PolicyAgent用並列学習の補助関数集"""

from ..ai.agents import EpisodeStep


def play_one_episode(env, agent_black, agent_white, policy_color="black"):
    """1エピソード(=1ゲーム)を実行し、学習ログを返す"""
    obs = env.reset()
    done = False

    policy_agent = agent_black if policy_color == "black" else agent_white
    start_log_len = len(policy_agent.episode_log)

    while not done:
        if env.current_player == 1:
            action = agent_black.get_action(obs, env)
        else:
            action = agent_white.get_action(obs, env)

        next_obs, reward, done, info = env.step(action)

        if policy_color == "black" and env.current_player == 2:
            policy_agent.record_reward(reward)
        elif policy_color == "white" and env.current_player == 1:
            policy_agent.record_reward(reward)

        obs = next_obs

    winner = info["winner"]
    turn_count = env.turn_count

    if (policy_color == "black" and winner == 2) or (
        policy_color == "white" and winner == 1
    ):
        policy_agent.record_reward(-1.0)
    elif winner == -1:
        policy_agent.record_reward(0.0)

    end_log_len = len(policy_agent.episode_log)
    episode_log = policy_agent.episode_log[start_log_len:end_log_len]

    return episode_log, winner, turn_count


def update_with_trajectories(agent, all_episodes):
    """収集済みエピソードをまとめて学習する"""
    import torch
    import torch.nn.functional as F

    all_states = []
    all_actions = []
    all_returns = []

    for episode_log in all_episodes:
        if not episode_log:
            continue

        is_obj = isinstance(episode_log[0], EpisodeStep)

        G = 0.0
        returns = []
        rewards = [step.reward if is_obj else step[2] for step in episode_log]
        for r in reversed(rewards):
            G = agent.gamma * G + r
            returns.insert(0, G)

        for idx, step in enumerate(episode_log):
            s = step.state if is_obj else step[0]
            a = step.action if is_obj else step[1]
            all_states.append(s)
            all_actions.append(a)
            all_returns.append(returns[idx])

    if len(all_states) == 0:
        return 0.0

    states_tensor = torch.cat(all_states, dim=0).to(agent.device)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long).to(agent.device)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(agent.device)

    logits = agent.model(states_tensor)
    log_probs = F.log_softmax(logits, dim=1)
    chosen_log_probs = log_probs[range(len(actions_tensor)), actions_tensor]

    loss = -(returns_tensor * chosen_log_probs).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    return loss.item()
