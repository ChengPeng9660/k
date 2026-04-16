import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch


LAG_1 = 8
FEATURE_DIM = 15
N_ACTIONS = 23
N_PAGES = 3


def make_action_profiles(rng):
    return rng.normal(0.0, 1.0, size=(N_PAGES, N_ACTIONS, FEATURE_DIM))


def build_feature(user_id, t, page, next_page, action, reward, latent, within_pre, within_next):
    pre_session = []
    for shift in range(LAG_1):
        drift = 0.05 * (t - shift)
        pre_session.append(latent + drift + np.random.normal(0.0, 0.12, FEATURE_DIM))
    return [
        torch.tensor(np.asarray(pre_session), dtype=torch.float32),
        torch.tensor(np.asarray(within_pre), dtype=torch.float32),
        torch.tensor(np.asarray(within_next), dtype=torch.float32),
        int(action),
        float(reward),
        int(page),
        int(next_page),
        int(user_id),
    ]


def generate_split_default(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    action_profiles = make_action_profiles(rng)

    features = []
    for user_id in range(num_users):
        latent = rng.normal(0.0, 1.0, FEATURE_DIM)
        preference = rng.normal(0.0, 0.8, N_ACTIONS)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))

        within_history = []
        for t in range(steps):
            context = latent + rng.normal(0.0, 0.25, FEATURE_DIM)
            logits = preference + action_profiles[page] @ context / FEATURE_DIM
            logits += rng.normal(0.0, 0.15, N_ACTIONS)
            action = int(np.argmax(logits))

            purchase_signal = logits[action] + 0.3 * context[0] - 0.15 * abs(page - 1)
            purchase_prob = 1.0 / (1.0 + np.exp(-purchase_signal))
            buy = rng.random() < purchase_prob
            reward = float(rng.integers(1, 11) * 100 if buy else 0)

            next_page = int((page + rng.choice([0, 1, 2], p=[0.55, 0.3, 0.15])) % N_PAGES)
            state_vec = (
                0.6 * context
                + 0.25 * action_profiles[page, action]
                + 0.15 * rng.normal(0.0, 1.0, FEATURE_DIM)
            )
            next_vec = (
                0.55 * context
                + 0.3 * action_profiles[next_page, action]
                + 0.15 * rng.normal(0.0, 1.0, FEATURE_DIM)
            )

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]

            features.append(
                build_feature(
                    user_id=user_id,
                    t=t,
                    page=page,
                    next_page=next_page,
                    action=action,
                    reward=reward,
                    latent=latent,
                    within_pre=within_pre,
                    within_next=within_next,
                )
            )

            within_history.append(state_vec)
            latent = 0.92 * latent + 0.08 * np.array(next_vec)
            page = next_page

    return features


def make_brqn_templates(rng):
    regime_templates = rng.normal(0.0, 0.7, size=(3, FEATURE_DIM))
    action_profiles = rng.normal(0.0, 0.55, size=(N_PAGES, N_ACTIONS, FEATURE_DIM))
    preferred = np.array(
        [
            [2, 7, 11],
            [4, 9, 15],
            [1, 13, 18],
        ]
    )
    bridge = np.array(
        [
            [7, 11, 2],
            [9, 15, 4],
            [13, 18, 1],
        ]
    )
    ordered_pair = np.array(
        [
            [[7, 2], [11, 7], [2, 11]],
            [[9, 4], [15, 9], [4, 15]],
            [[13, 1], [18, 13], [1, 18]],
        ]
    )
    return regime_templates, action_profiles, preferred, bridge, ordered_pair


def generate_split_brqn_favor(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_templates, action_profiles, preferred, bridge, ordered_pair = make_brqn_templates(rng)

    features = []
    for user_id in range(num_users):
        user_core = rng.normal(0.0, 0.8, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        hidden_regime = int(rng.integers(0, 3))
        previous_actions = [int(rng.integers(0, N_ACTIONS)), int(rng.integers(0, N_ACTIONS))]
        within_history = []

        for t in range(steps):
            long_history = np.mean(within_history[-6:], axis=0) if within_history else np.zeros(FEATURE_DIM)
            latent = 0.55 * user_core + 0.75 * regime_templates[hidden_regime] + 0.35 * long_history
            observed_context = 0.55 * user_core + 0.05 * regime_templates[hidden_regime] + rng.normal(0.0, 0.55, FEATURE_DIM)

            preferred_action = int(preferred[hidden_regime, page])
            bridge_action = int(bridge[hidden_regime, page])
            exact_pair = ordered_pair[hidden_regime, page]
            ordered_match = previous_actions[-2] == int(exact_pair[0]) and previous_actions[-1] == int(exact_pair[1])
            bridge_match = previous_actions[-1] == bridge_action
            logits = action_profiles[page] @ observed_context / FEATURE_DIM
            logits += rng.normal(0.0, 0.45, N_ACTIONS)
            logits[preferred_action] += 0.75
            logits[bridge_action] += 0.6
            if bridge_match:
                logits[preferred_action] += 0.45
            if ordered_match:
                logits[preferred_action] += 1.25

            action = int(rng.choice(N_ACTIONS, p=np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()))

            hidden_bonus = 1.1 if action == preferred_action else -0.35
            sequence_bonus = 2.4 if (ordered_match and action == preferred_action) else 0.0
            bridge_sequence_bonus = 0.75 if (bridge_match and action == preferred_action) else 0.0
            exploratory_bonus = 0.9 if action == bridge_action else 0.0
            jackpot = 1.0 if (action == preferred_action and rng.random() < 0.15 + 0.40 * ordered_match) else 0.0
            purchase_signal = (
                0.55 * logits[action]
                + hidden_bonus
                + sequence_bonus
                + bridge_sequence_bonus
                + exploratory_bonus
                + 0.12 * observed_context[0]
                - 0.12 * abs(page - hidden_regime)
            )
            purchase_prob = 1.0 / (1.0 + np.exp(-purchase_signal))
            buy = rng.random() < purchase_prob

            if buy:
                base_reward = float(rng.integers(2, 7) * 100)
                if jackpot:
                    base_reward += float(rng.integers(8, 14) * 100)
                reward = base_reward
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20])) % N_PAGES)
            state_vec = (
                0.20 * observed_context
                + 0.50 * latent
                + 0.25 * action_profiles[page, action]
                + 0.15 * regime_templates[hidden_regime]
                + rng.normal(0.0, 0.10, FEATURE_DIM)
            )
            next_vec = (
                0.20 * observed_context
                + 0.50 * regime_templates[(hidden_regime + (1 if action == bridge_action else 0)) % 3]
                + 0.30 * action_profiles[next_page, action]
                + rng.normal(0.0, 0.10, FEATURE_DIM)
            )

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                build_feature(
                    user_id=user_id,
                    t=t,
                    page=page,
                    next_page=next_page,
                    action=action,
                    reward=reward,
                    latent=latent,
                    within_pre=within_pre,
                    within_next=within_next,
                )
            )

            within_history.append(state_vec)
            previous_actions.append(action)
            previous_actions = previous_actions[-2:]
            if action == bridge_action and rng.random() < 0.7:
                hidden_regime = (hidden_regime + 1) % 3
            elif rng.random() < 0.05:
                hidden_regime = int(rng.integers(0, 3))
            user_core = 0.95 * user_core + 0.05 * np.array(next_vec)
            page = next_page

    return features


def generate_split_brqn_linear_sparse(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    action_profiles = rng.normal(0.0, 0.45, size=(N_ACTIONS, FEATURE_DIM))
    bridge_actions = np.array([3, 8, 14, 19])

    features = []
    for user_id in range(num_users):
        latent = rng.normal(0.0, 1.0, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_action = int(rng.integers(0, N_ACTIONS))
        hidden_gate = int(rng.integers(0, 4))

        for t in range(steps):
            long_history = np.mean(within_history[-5:], axis=0) if within_history else np.zeros(FEATURE_DIM)
            latent = 0.82 * latent + 0.18 * long_history + rng.normal(0.0, 0.08, FEATURE_DIM)
            observed_context = 0.25 * latent + rng.normal(0.0, 0.75, FEATURE_DIM)

            sparse_probe = bridge_actions[hidden_gate]
            preferred_action = int(np.argmax(action_profiles @ latent))
            if prev_action == sparse_probe and rng.random() < 0.75:
                hidden_gate = (hidden_gate + 1) % len(bridge_actions)
            gated_optimal = (preferred_action + hidden_gate) % N_ACTIONS

            logits = action_profiles @ observed_context
            logits = logits / FEATURE_DIM + rng.normal(0.0, 0.55, N_ACTIONS)
            logits[gated_optimal] += 0.55
            logits[sparse_probe] += 0.35
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            expected_value = float(np.dot(action_profiles[action], latent))
            gated_bonus = 2.8 if action == gated_optimal else -0.55
            sequence_bonus = 1.9 if prev_action == sparse_probe and action == gated_optimal else 0.0
            reward_noise_scale = 0.35 if action == gated_optimal else 1.1
            noisy_signal = expected_value + gated_bonus + sequence_bonus + rng.normal(0.0, reward_noise_scale)
            buy = noisy_signal > 0.75

            if buy:
                reward = float(max(1, int((noisy_signal + 1.5) * 2)) * 100)
                if action == gated_optimal and rng.random() < 0.22:
                    reward += float(rng.integers(6, 11) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15])) % N_PAGES)
            state_vec = 0.55 * latent + 0.20 * observed_context + 0.25 * action_profiles[action] + rng.normal(0.0, 0.10, FEATURE_DIM)
            next_vec = 0.60 * latent + 0.25 * action_profiles[gated_optimal] + 0.15 * rng.normal(0.0, 1.0, FEATURE_DIM)

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                build_feature(
                    user_id=user_id,
                    t=t,
                    page=page,
                    next_page=next_page,
                    action=action,
                    reward=reward,
                    latent=latent,
                    within_pre=within_pre,
                    within_next=within_next,
                )
            )

            within_history.append(state_vec)
            prev_action = action
            page = next_page

    return features


def generate_split_brqn_mechanism(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = rng.normal(0.0, 0.9, size=(4, FEATURE_DIM))
    action_profiles = rng.normal(0.0, 0.35, size=(N_ACTIONS, FEATURE_DIM))
    bridge_actions = np.array([2, 7, 12, 18])
    regime_targets = np.array([5, 10, 15, 20])

    features = []
    for user_id in range(num_users):
        base_latent = rng.normal(0.0, 0.7, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_actions = [int(rng.integers(0, N_ACTIONS)), int(rng.integers(0, N_ACTIONS))]
        regime = int(rng.integers(0, 4))

        for t in range(steps):
            hist_short = np.mean(within_history[-3:], axis=0) if within_history else np.zeros(FEATURE_DIM)
            hist_long = np.mean(within_history[-8:], axis=0) if within_history else np.zeros(FEATURE_DIM)
            latent = 0.35 * base_latent + 0.95 * regime_vectors[regime] + 0.55 * hist_long
            observed_context = 0.12 * latent + 0.10 * hist_short + rng.normal(0.0, 0.95, FEATURE_DIM)

            bridge = int(bridge_actions[regime])
            target = int(regime_targets[regime])
            bridge_ready = prev_actions[-1] == bridge
            pair_ready = prev_actions[-2] == bridge and prev_actions[-1] == target
            cycle_target = int((target + page) % N_ACTIONS)

            logits = action_profiles @ observed_context / FEATURE_DIM
            logits += rng.normal(0.0, 0.60, N_ACTIONS)
            logits[bridge] += 0.45
            logits[target] += 0.25
            logits[cycle_target] += 0.20
            if bridge_ready:
                logits[target] += 0.55
            if pair_ready:
                logits[target] += 0.95
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            linear_value = float(np.dot(action_profiles[action], latent) / FEATURE_DIM)
            action_match = 1.0 if action == target else 0.0
            bridge_bonus = 0.8 if (bridge_ready and action == target) else 0.0
            pair_bonus = 2.4 if (pair_ready and action == target) else 0.0
            page_bonus = 0.5 if action == cycle_target else -0.1
            jackpot_prob = 0.03 + 0.32 * action_match + 0.30 * float(pair_ready and action == target)
            reward_noise = 0.18 if action == target else 0.95
            buy_signal = linear_value + 1.6 * action_match + bridge_bonus + pair_bonus + page_bonus + rng.normal(0.0, reward_noise)
            buy = buy_signal > 0.95

            if buy:
                reward = float(max(1, int((buy_signal + 0.9) * 2)) * 100)
                if rng.random() < jackpot_prob:
                    reward += float(rng.integers(8, 15) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.30, 0.55, 0.15])) % N_PAGES)
            if action == bridge and rng.random() < 0.85:
                next_regime = (regime + 1) % len(regime_vectors)
            elif pair_ready and action == target and rng.random() < 0.70:
                next_regime = (regime + 2) % len(regime_vectors)
            elif rng.random() < 0.03:
                next_regime = int(rng.integers(0, len(regime_vectors)))
            else:
                next_regime = regime

            state_vec = (
                0.62 * latent
                + 0.10 * observed_context
                + 0.18 * action_profiles[action]
                + 0.10 * regime_vectors[regime]
                + rng.normal(0.0, 0.08, FEATURE_DIM)
            )
            next_vec = (
                0.70 * regime_vectors[next_regime]
                + 0.18 * action_profiles[target]
                + 0.12 * hist_short
                + rng.normal(0.0, 0.08, FEATURE_DIM)
            )

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                build_feature(
                    user_id=user_id,
                    t=t,
                    page=page,
                    next_page=next_page,
                    action=action,
                    reward=reward,
                    latent=latent,
                    within_pre=within_pre,
                    within_next=within_next,
                )
            )

            within_history.append(state_vec)
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            base_latent = 0.94 * base_latent + 0.06 * np.array(next_vec)
            page = next_page

    return features


def generate_split_brqn_order_uncertainty(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = rng.normal(0.0, 0.75, size=(3, FEATURE_DIM))
    action_profiles = rng.normal(0.0, 0.22, size=(N_ACTIONS, FEATURE_DIM))
    symbol_vectors = np.zeros((4, FEATURE_DIM))
    symbol_vectors[0, 0] = 1.0
    symbol_vectors[1, 1] = 1.0
    symbol_vectors[2, 0] = 1.0
    symbol_vectors[2, 1] = 1.0
    symbol_vectors[3, 2] = 1.0
    bridge_actions = np.array([2, 7, 12])
    target_actions = np.array([5, 10, 15])

    features = []
    for user_id in range(num_users):
        base_latent = rng.normal(0.0, 0.45, FEATURE_DIM)
        regime = int(rng.integers(0, 3))
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_symbols = [0, 1]
        prev_actions = [int(rng.integers(0, N_ACTIONS)), int(rng.integers(0, N_ACTIONS))]

        for t in range(steps):
            current_symbol = int(rng.integers(0, 2))
            long_hist = np.mean(within_history[-8:], axis=0) if within_history else np.zeros(FEATURE_DIM)
            latent = 0.25 * base_latent + 0.90 * regime_vectors[regime] + 0.40 * long_hist

            # Sequence patterns AB and BA share the same sum, so DQN/XGBoost lose this order signal.
            pair_ab = prev_symbols[-1] == 0 and current_symbol == 1
            pair_ba = prev_symbols[-1] == 1 and current_symbol == 0
            bridge = int(bridge_actions[regime])
            target = int(target_actions[regime])
            order_ready = pair_ab and prev_actions[-1] == bridge
            decoy_ready = pair_ba and prev_actions[-1] == bridge

            observed_context = (
                0.08 * latent
                + 0.22 * symbol_vectors[current_symbol]
                + 0.10 * symbol_vectors[prev_symbols[-1]]
                + rng.normal(0.0, 0.95, FEATURE_DIM)
            )
            logits = action_profiles @ observed_context / FEATURE_DIM
            logits += rng.normal(0.0, 0.65, N_ACTIONS)
            logits[bridge] += 0.50
            logits[target] += 0.15
            if pair_ab:
                logits[target] += 0.25
            if order_ready:
                logits[target] += 0.95
            if decoy_ready:
                logits[target] -= 0.20
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            linear_value = float(np.dot(action_profiles[action], latent) / FEATURE_DIM)
            match_bonus = 1.8 if action == target else -0.35
            order_bonus = 2.8 if (order_ready and action == target) else 0.0
            decoy_penalty = -1.1 if (decoy_ready and action == target) else 0.0
            bridge_value = 0.65 if action == bridge else 0.0
            reward_noise = 0.15 if action == target else 0.95
            signal = linear_value + match_bonus + order_bonus + decoy_penalty + bridge_value + rng.normal(0.0, reward_noise)
            buy = signal > 1.15

            if buy:
                reward = float(max(1, int((signal + 1.2) * 2)) * 100)
                if order_ready and action == target and rng.random() < 0.40:
                    reward += float(rng.integers(7, 13) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.34, 0.48, 0.18])) % N_PAGES)
            if action == bridge and pair_ab and rng.random() < 0.80:
                next_regime = (regime + 1) % 3
            elif order_ready and action == target and rng.random() < 0.55:
                next_regime = regime
            elif rng.random() < 0.04:
                next_regime = int(rng.integers(0, 3))
            else:
                next_regime = regime

            state_vec = (
                0.55 * latent
                + 0.16 * symbol_vectors[current_symbol]
                + 0.10 * symbol_vectors[prev_symbols[-1]]
                + 0.19 * action_profiles[action]
                + rng.normal(0.0, 0.06, FEATURE_DIM)
            )
            next_symbol = int(rng.integers(0, 2))
            next_vec = (
                0.62 * regime_vectors[next_regime]
                + 0.16 * symbol_vectors[current_symbol]
                + 0.16 * symbol_vectors[next_symbol]
                + 0.06 * rng.normal(0.0, 1.0, FEATURE_DIM)
            )

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                build_feature(
                    user_id=user_id,
                    t=t,
                    page=page,
                    next_page=next_page,
                    action=action,
                    reward=reward,
                    latent=latent,
                    within_pre=within_pre,
                    within_next=within_next,
                )
            )

            within_history.append(state_vec)
            prev_symbols.append(current_symbol)
            prev_symbols = prev_symbols[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            base_latent = 0.95 * base_latent + 0.05 * np.array(next_vec)
            page = next_page

    return features


def generate_split_brqn_horizon_support(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = rng.normal(0.0, 0.65, size=(3, FEATURE_DIM))
    action_profiles = rng.normal(0.0, 0.12, size=(N_ACTIONS, FEATURE_DIM))
    token_vectors = np.zeros((2, FEATURE_DIM))
    token_vectors[0, 0] = 1.0
    token_vectors[1, 1] = 1.0
    bridge_actions = np.array([2, 7, 12])
    target_actions = np.array([5, 10, 15])
    decoy_actions = np.array([6, 11, 16])

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        base_latent = rng.normal(0.0, 0.35, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_tokens = [0, 1]
        prev_actions = [bridge_actions[regime], target_actions[regime]]

        for t in range(steps):
            token = int(rng.integers(0, 2))
            hist_long = np.mean(within_history[-8:], axis=0) if within_history else np.zeros(FEATURE_DIM)
            latent = 0.20 * base_latent + 0.95 * regime_vectors[regime] + 0.30 * hist_long

            bridge = int(bridge_actions[regime])
            target = int(target_actions[regime])
            decoy = int(decoy_actions[regime])
            order_ready = prev_tokens[-1] == 0 and token == 1 and prev_actions[-1] == bridge
            wrong_order = prev_tokens[-1] == 1 and token == 0 and prev_actions[-1] == bridge

            observed_context = (
                0.04 * latent
                + 0.18 * token_vectors[token]
                + 0.18 * token_vectors[prev_tokens[-1]]
                + rng.normal(0.0, 1.00, FEATURE_DIM)
            )

            probs = np.full(N_ACTIONS, 0.002)
            probs[bridge] = 0.42
            probs[target] = 0.34
            probs[decoy] = 0.16
            if order_ready:
                probs[target] = 0.72
                probs[bridge] = 0.18
                probs[decoy] = 0.06
            elif wrong_order:
                probs[decoy] = 0.46
                probs[target] = 0.18
                probs[bridge] = 0.28
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            linear_value = float(np.dot(action_profiles[action], latent) / FEATURE_DIM)
            base_bonus = 0.3 if action == bridge else -0.2
            target_bonus = 1.2 if action == target else -0.35
            order_bonus = 4.0 if (order_ready and action == target) else 0.0
            wrong_penalty = -2.0 if (wrong_order and action == target) else 0.0
            decoy_bonus = 0.4 if (wrong_order and action == decoy) else 0.0
            noise_scale = 0.12 if action == target else 0.55
            signal = linear_value + base_bonus + target_bonus + order_bonus + wrong_penalty + decoy_bonus + rng.normal(0.0, noise_scale)
            buy = signal > 1.25

            if buy:
                reward = float(max(1, int((signal + 0.8) * 2)) * 100)
                if order_ready and action == target and rng.random() < 0.55:
                    reward += float(rng.integers(10, 18) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20])) % N_PAGES)
            next_regime = regime
            if action == target and order_ready and rng.random() < 0.60:
                next_regime = (regime + 1) % 3
            elif action == decoy and wrong_order and rng.random() < 0.30:
                next_regime = (regime + 1) % 3

            state_vec = (
                0.58 * regime_vectors[regime]
                + 0.14 * token_vectors[token]
                + 0.14 * token_vectors[prev_tokens[-1]]
                + 0.14 * action_profiles[action]
                + rng.normal(0.0, 0.05, FEATURE_DIM)
            )
            next_token = int(rng.integers(0, 2))
            next_vec = (
                0.64 * regime_vectors[next_regime]
                + 0.18 * token_vectors[token]
                + 0.12 * token_vectors[next_token]
                + 0.06 * rng.normal(0.0, 1.0, FEATURE_DIM)
            )

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                build_feature(
                    user_id=user_id,
                    t=t,
                    page=page,
                    next_page=next_page,
                    action=action,
                    reward=reward,
                    latent=latent,
                    within_pre=within_pre,
                    within_next=within_next,
                )
            )

            within_history.append(state_vec)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            base_latent = 0.97 * base_latent + 0.03 * np.array(next_vec)
            page = next_page

    return features


def generate_split_brqn_presession_linear(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = rng.normal(0.0, 0.85, size=(3, FEATURE_DIM))
    action_weights = rng.normal(0.0, 0.25, size=(N_ACTIONS, FEATURE_DIM))
    bridge_actions = np.array([2, 7, 12])
    target_actions = np.array([5, 10, 15])
    decoy_actions = np.array([6, 11, 16])
    token_a = np.zeros(FEATURE_DIM)
    token_b = np.zeros(FEATURE_DIM)
    token_a[0] = 1.0
    token_b[0] = -1.0

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        latent = regime_vectors[regime] + rng.normal(0.0, 0.10, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_tokens = [0, 1]
        prev_actions = [bridge_actions[regime], target_actions[regime]]

        for t in range(steps):
            bridge = int(bridge_actions[regime])
            target = int(target_actions[regime])
            decoy = int(decoy_actions[regime])
            token = int(rng.integers(0, 2))
            order_ready = prev_tokens[-1] == 0 and token == 1 and prev_actions[-1] == bridge
            wrong_order = prev_tokens[-1] == 1 and token == 0 and prev_actions[-1] == bridge

            probs = np.full(N_ACTIONS, 0.001)
            probs[bridge] = 0.46
            probs[target] = 0.30
            probs[decoy] = 0.18
            if order_ready:
                probs[target] = 0.74
                probs[bridge] = 0.16
                probs[decoy] = 0.06
            elif wrong_order:
                probs[decoy] = 0.46
                probs[target] = 0.14
                probs[bridge] = 0.28
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            linear_value = float(np.dot(action_weights[action], latent))
            target_bonus = 1.6 if action == target else -0.25
            order_bonus = 3.8 if (order_ready and action == target) else 0.0
            wrong_penalty = -2.1 if (wrong_order and action == target) else 0.0
            decoy_bonus = 0.6 if (wrong_order and action == decoy) else 0.0
            noise_scale = 0.12 if action == target else 0.75
            signal = linear_value + target_bonus + order_bonus + wrong_penalty + decoy_bonus + rng.normal(0.0, noise_scale)
            buy = signal > 1.20

            if buy:
                reward = float(max(1, int((signal + 1.0) * 2)) * 100)
                if order_ready and action == target and rng.random() < 0.50:
                    reward += float(rng.integers(8, 16) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.36, 0.44, 0.20])) % N_PAGES)
            next_regime = regime
            if order_ready and action == target and rng.random() < 0.55:
                next_regime = (regime + 1) % 3

            # Make within-session sums nearly uninformative for DQN/XGBoost.
            current_token_vec = token_a if token == 0 else token_b
            prev_token_vec = token_a if prev_tokens[-1] == 0 else token_b
            state_vec = 0.04 * current_token_vec + 0.04 * prev_token_vec + rng.normal(0.0, 0.01, FEATURE_DIM)
            next_token = int(rng.integers(0, 2))
            next_token_vec = token_a if next_token == 0 else token_b
            next_vec = 0.04 * current_token_vec + 0.04 * next_token_vec + rng.normal(0.0, 0.01, FEATURE_DIM)

            pre_session = []
            for shift in range(LAG_1):
                pre_session.append(latent + 0.03 * shift + rng.normal(0.0, 0.03, FEATURE_DIM))

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                [
                    torch.tensor(np.asarray(pre_session), dtype=torch.float32),
                    torch.tensor(np.asarray(within_pre), dtype=torch.float32),
                    torch.tensor(np.asarray(within_next), dtype=torch.float32),
                    int(action),
                    float(reward),
                    int(page),
                    int(next_page),
                    int(user_id),
                ]
            )

            within_history.append(state_vec)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            latent = 0.96 * latent + 0.04 * regime_vectors[next_regime]
            page = next_page

    return features


def generate_split_brqn_regime_action(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = np.zeros((3, FEATURE_DIM))
    regime_vectors[0, 0] = 1.0
    regime_vectors[1, 1] = 1.0
    regime_vectors[2, 2] = 1.0
    bridge_actions = np.array([2, 7, 12])
    target_actions = np.array([5, 10, 15])
    decoy_actions = np.array([6, 11, 16])

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        latent = regime_vectors[regime] + rng.normal(0.0, 0.04, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_tokens = [0, 1]
        prev_actions = [bridge_actions[regime], target_actions[regime]]

        for t in range(steps):
            bridge = int(bridge_actions[regime])
            target = int(target_actions[regime])
            decoy = int(decoy_actions[regime])
            token = int(rng.integers(0, 2))
            order_ready = prev_tokens[-1] == 0 and token == 1 and prev_actions[-1] == bridge
            wrong_order = prev_tokens[-1] == 1 and token == 0 and prev_actions[-1] == bridge

            probs = np.full(N_ACTIONS, 0.0005)
            probs[bridge] = 0.44
            probs[target] = 0.30
            probs[decoy] = 0.18
            if order_ready:
                probs[target] = 0.70
                probs[bridge] = 0.18
                probs[decoy] = 0.08
            elif wrong_order:
                probs[decoy] = 0.44
                probs[target] = 0.16
                probs[bridge] = 0.28
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            # Keep wrong/correct target actions globally symmetric across regimes.
            correct_target = action == target
            correct_bridge = action == bridge
            correct_decoy = action == decoy
            signal = -0.6
            if correct_bridge:
                signal += 0.4
            if correct_target:
                signal += 0.5
            if order_ready and correct_target:
                signal += 3.8
            if wrong_order and correct_target:
                signal -= 2.4
            if wrong_order and correct_decoy:
                signal += 0.8
            signal += rng.normal(0.0, 0.18 if correct_target else 0.60)
            buy = signal > 1.15

            if buy:
                reward = 300.0
                if order_ready and correct_target:
                    reward += float(rng.integers(10, 18) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.38, 0.42, 0.20])) % N_PAGES)
            next_regime = regime
            if order_ready and correct_target and rng.random() < 0.60:
                next_regime = (regime + 1) % 3

            pre_session = []
            for shift in range(LAG_1):
                pre_session.append(latent + 0.02 * shift + rng.normal(0.0, 0.015, FEATURE_DIM))

            # Deliberately uninformative under summation.
            state_vec = rng.normal(0.0, 0.003, FEATURE_DIM)
            next_vec = rng.normal(0.0, 0.003, FEATURE_DIM)

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                [
                    torch.tensor(np.asarray(pre_session), dtype=torch.float32),
                    torch.tensor(np.asarray(within_pre), dtype=torch.float32),
                    torch.tensor(np.asarray(within_next), dtype=torch.float32),
                    int(action),
                    float(reward),
                    int(page),
                    int(next_page),
                    int(user_id),
                ]
            )

            within_history.append(state_vec)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            latent = 0.98 * latent + 0.02 * regime_vectors[next_regime]
            page = next_page

    return features


def generate_split_brqn_user_mapping(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = np.zeros((3, FEATURE_DIM))
    regime_vectors[0, 0] = 1.0
    regime_vectors[1, 1] = 1.0
    regime_vectors[2, 2] = 1.0
    action_banks = np.array(
        [
            [2, 5, 6],
            [7, 10, 11],
            [12, 15, 16],
        ]
    )

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        latent = regime_vectors[regime] + rng.normal(0.0, 0.03, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_tokens = [0, 1]
        user_shift = int(rng.integers(0, 3))
        perm = np.roll(np.array([0, 1, 2]), user_shift)
        prev_actions = [action_banks[0, perm[0]], action_banks[0, perm[1]]]

        for t in range(steps):
            bridge = int(action_banks[regime, perm[0]])
            target = int(action_banks[regime, perm[1]])
            decoy = int(action_banks[regime, perm[2]])
            token = int(rng.integers(0, 2))
            order_ready = prev_tokens[-1] == 0 and token == 1 and prev_actions[-1] == bridge
            wrong_order = prev_tokens[-1] == 1 and token == 0 and prev_actions[-1] == bridge

            probs = np.full(N_ACTIONS, 0.0005)
            probs[bridge] = 0.42
            probs[target] = 0.28
            probs[decoy] = 0.16
            if order_ready:
                probs[target] = 0.68
                probs[bridge] = 0.16
                probs[decoy] = 0.10
            elif wrong_order:
                probs[decoy] = 0.42
                probs[target] = 0.14
                probs[bridge] = 0.26
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            signal = -0.7
            if action == bridge:
                signal += 0.35
            if action == target:
                signal += 0.45
            if order_ready and action == target:
                signal += 3.8
            if wrong_order and action == target:
                signal -= 2.3
            if wrong_order and action == decoy:
                signal += 0.9
            signal += rng.normal(0.0, 0.16 if action == target else 0.55)
            buy = signal > 1.2

            if buy:
                reward = 300.0
                if order_ready and action == target:
                    reward += float(rng.integers(10, 18) * 100)
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.38, 0.42, 0.20])) % N_PAGES)
            next_regime = regime
            if order_ready and action == target and rng.random() < 0.60:
                next_regime = (regime + 1) % 3

            pre_session = []
            for shift in range(LAG_1):
                snapshot = latent + 0.02 * shift + rng.normal(0.0, 0.015, FEATURE_DIM)
                snapshot[3] = float(user_shift) / 2.0
                snapshot[4] = float(perm[0])
                snapshot[5] = float(perm[1])
                snapshot[6] = float(perm[2])
                pre_session.append(snapshot)

            state_vec = rng.normal(0.0, 0.003, FEATURE_DIM)
            next_vec = rng.normal(0.0, 0.003, FEATURE_DIM)

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                [
                    torch.tensor(np.asarray(pre_session), dtype=torch.float32),
                    torch.tensor(np.asarray(within_pre), dtype=torch.float32),
                    torch.tensor(np.asarray(within_next), dtype=torch.float32),
                    int(action),
                    float(reward),
                    int(page),
                    int(next_page),
                    int(user_id),
                ]
            )

            within_history.append(state_vec)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            latent = 0.98 * latent + 0.02 * regime_vectors[next_regime]
            page = next_page

    return features


def generate_split_brqn_user_mapping_horizon(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = np.zeros((3, FEATURE_DIM))
    regime_vectors[0, 0] = 1.0
    regime_vectors[1, 1] = 1.0
    regime_vectors[2, 2] = 1.0
    action_banks = np.array(
        [
            [2, 5, 6],
            [7, 10, 11],
            [12, 15, 16],
        ]
    )
    all_perms = [
        np.array([0, 1, 2]),
        np.array([0, 2, 1]),
        np.array([1, 0, 2]),
        np.array([1, 2, 0]),
        np.array([2, 0, 1]),
        np.array([2, 1, 0]),
    ]

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        latent = regime_vectors[regime] + rng.normal(0.0, 0.025, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_tokens = [0, 1]
        perm = all_perms[int(rng.integers(0, len(all_perms)))]
        perm_id = int(np.where([(perm == p).all() for p in all_perms])[0][0])
        prev_actions = [action_banks[0, perm[0]], action_banks[0, perm[1]]]
        bonus_counter = 0

        for t in range(steps):
            bridge = int(action_banks[regime, perm[0]])
            target = int(action_banks[regime, perm[1]])
            decoy = int(action_banks[regime, perm[2]])
            token = int(rng.integers(0, 2))
            order_ready = prev_tokens[-1] == 0 and token == 1 and prev_actions[-1] == bridge
            wrong_order = prev_tokens[-1] == 1 and token == 0 and prev_actions[-1] == bridge

            probs = np.full(N_ACTIONS, 0.0005)
            probs[bridge] = 0.34
            probs[target] = 0.24
            probs[decoy] = 0.10
            if order_ready:
                probs[target] = 0.84
                probs[bridge] = 0.10
                probs[decoy] = 0.03
            elif bonus_counter > 0:
                probs[target] = 0.76
                probs[bridge] = 0.10
                probs[decoy] = 0.04
            elif wrong_order:
                probs[decoy] = 0.36
                probs[target] = 0.10
                probs[bridge] = 0.22
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            signal = -1.0
            if action == bridge:
                signal += 0.20
            if action == target:
                signal += 0.65
            if order_ready and action == target:
                signal += 4.4
            if bonus_counter > 0 and action == target:
                signal += 2.1
            if wrong_order and action == target:
                signal -= 2.4
            if wrong_order and action == decoy:
                signal += 0.50
            signal += rng.normal(0.0, 0.10 if action == target else 0.50)
            buy = signal > 1.0

            if buy:
                if order_ready and action == target:
                    reward = float(rng.integers(10, 15) * 100)
                elif bonus_counter > 0 and action == target:
                    reward = float(rng.integers(7, 11) * 100)
                elif action == target:
                    reward = float(rng.integers(3, 6) * 100)
                else:
                    reward = 100.0
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.40, 0.40, 0.20])) % N_PAGES)
            next_regime = regime
            next_bonus_counter = max(0, bonus_counter - 1)
            if order_ready and action == target:
                next_regime = (regime + 1) % 3
                next_bonus_counter = 2

            pre_session = []
            for shift in range(LAG_1):
                snapshot = latent + 0.015 * shift + rng.normal(0.0, 0.012, FEATURE_DIM)
                snapshot[3] = float(perm_id) / 5.0
                snapshot[4] = float(perm[0])
                snapshot[5] = float(perm[1])
                snapshot[6] = float(perm[2])
                snapshot[7] = float(bonus_counter) / 2.0
                snapshot[8] = float(regime)
                pre_session.append(snapshot)

            state_vec = rng.normal(0.0, 0.003, FEATURE_DIM)
            next_vec = rng.normal(0.0, 0.003, FEATURE_DIM)

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                [
                    torch.tensor(np.asarray(pre_session), dtype=torch.float32),
                    torch.tensor(np.asarray(within_pre), dtype=torch.float32),
                    torch.tensor(np.asarray(within_next), dtype=torch.float32),
                    int(action),
                    float(reward),
                    int(page),
                    int(next_page),
                    int(user_id),
                ]
            )

            within_history.append(state_vec)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            bonus_counter = next_bonus_counter
            latent = 0.985 * latent + 0.015 * regime_vectors[next_regime]
            page = next_page

    return features


def generate_split_brqn_user_mapping_longbonus(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = np.zeros((3, FEATURE_DIM))
    regime_vectors[0, 0] = 1.0
    regime_vectors[1, 1] = 1.0
    regime_vectors[2, 2] = 1.0
    action_banks = np.array(
        [
            [2, 5, 6],
            [7, 10, 11],
            [12, 15, 16],
        ]
    )
    all_perms = [
        np.array([0, 1, 2]),
        np.array([0, 2, 1]),
        np.array([1, 0, 2]),
        np.array([1, 2, 0]),
        np.array([2, 0, 1]),
        np.array([2, 1, 0]),
    ]

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        latent = regime_vectors[regime] + rng.normal(0.0, 0.02, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_tokens = [0, 1]
        perm = all_perms[int(rng.integers(0, len(all_perms)))]
        perm_id = int(np.where([(perm == p).all() for p in all_perms])[0][0])
        prev_actions = [action_banks[0, perm[0]], action_banks[0, perm[1]]]
        bonus_counter = 0

        for t in range(steps):
            bridge = int(action_banks[regime, perm[0]])
            target = int(action_banks[regime, perm[1]])
            decoy = int(action_banks[regime, perm[2]])
            token = int(rng.integers(0, 2))
            order_ready = prev_tokens[-1] == 0 and token == 1 and prev_actions[-1] == bridge
            wrong_order = prev_tokens[-1] == 1 and token == 0 and prev_actions[-1] == bridge

            probs = np.full(N_ACTIONS, 0.0005)
            probs[bridge] = 0.30
            probs[target] = 0.20
            probs[decoy] = 0.08
            if order_ready:
                probs[target] = 0.88
                probs[bridge] = 0.08
                probs[decoy] = 0.02
            elif bonus_counter > 0:
                probs[target] = 0.83
                probs[bridge] = 0.07
                probs[decoy] = 0.03
            elif wrong_order:
                probs[decoy] = 0.34
                probs[target] = 0.08
                probs[bridge] = 0.20
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            signal = -1.2
            if action == bridge:
                signal += 0.15
            if action == target:
                signal += 0.75
            if order_ready and action == target:
                signal += 4.8
            if bonus_counter > 0 and action == target:
                signal += 3.0
            if wrong_order and action == target:
                signal -= 2.8
            if wrong_order and action == decoy:
                signal += 0.45
            signal += rng.normal(0.0, 0.08 if action == target else 0.55)
            buy = signal > 0.95

            if buy:
                if order_ready and action == target:
                    reward = float(rng.integers(16, 22) * 100)
                elif bonus_counter > 0 and action == target:
                    reward = float(rng.integers(12, 18) * 100)
                elif action == target:
                    reward = float(rng.integers(4, 7) * 100)
                else:
                    reward = 100.0
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.42, 0.38, 0.20])) % N_PAGES)
            next_regime = regime
            next_bonus_counter = max(0, bonus_counter - 1)
            if order_ready and action == target:
                next_regime = (regime + 1) % 3
                next_bonus_counter = 5

            pre_session = []
            for shift in range(LAG_1):
                snapshot = latent + 0.012 * shift + rng.normal(0.0, 0.010, FEATURE_DIM)
                snapshot[3] = float(perm_id) / 5.0
                snapshot[4] = float(perm[0])
                snapshot[5] = float(perm[1])
                snapshot[6] = float(perm[2])
                snapshot[7] = float(min(bonus_counter, 5)) / 5.0
                snapshot[8] = float(regime)
                pre_session.append(snapshot)

            state_vec = rng.normal(0.0, 0.003, FEATURE_DIM)
            next_vec = rng.normal(0.0, 0.003, FEATURE_DIM)

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                [
                    torch.tensor(np.asarray(pre_session), dtype=torch.float32),
                    torch.tensor(np.asarray(within_pre), dtype=torch.float32),
                    torch.tensor(np.asarray(within_next), dtype=torch.float32),
                    int(action),
                    float(reward),
                    int(page),
                    int(next_page),
                    int(user_id),
                ]
            )

            within_history.append(state_vec)
            prev_tokens.append(token)
            prev_tokens = prev_tokens[-2:]
            prev_actions.append(action)
            prev_actions = prev_actions[-2:]
            regime = next_regime
            bonus_counter = next_bonus_counter
            latent = 0.988 * latent + 0.012 * regime_vectors[next_regime]
            page = next_page

    return features


def generate_split_brqn_simple(num_users, min_steps, max_steps, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    regime_vectors = np.zeros((3, FEATURE_DIM))
    regime_vectors[0, 0] = 1.0
    regime_vectors[1, 1] = 1.0
    regime_vectors[2, 2] = 1.0
    action_banks = np.array(
        [
            [2, 5, 6],
            [7, 10, 11],
            [12, 15, 16],
        ]
    )

    features = []
    for user_id in range(num_users):
        regime = int(rng.integers(0, 3))
        user_type = int(rng.integers(0, 3))
        bridge_idx = user_type
        target_idx = (user_type + 1) % 3
        decoy_idx = (user_type + 2) % 3
        latent = regime_vectors[regime] + rng.normal(0.0, 0.02, FEATURE_DIM)
        page = int(rng.integers(0, N_PAGES))
        steps = int(rng.integers(min_steps, max_steps + 1))
        within_history = []
        prev_token = 0
        prev_action = int(action_banks[regime, bridge_idx])
        carryover = 0

        for t in range(steps):
            bridge = int(action_banks[regime, bridge_idx])
            target = int(action_banks[regime, target_idx])
            decoy = int(action_banks[regime, decoy_idx])
            token = int(rng.integers(0, 2))
            order_ready = prev_token == 0 and token == 1 and prev_action == bridge
            wrong_order = prev_token == 1 and token == 0 and prev_action == bridge

            probs = np.full(N_ACTIONS, 0.0005)
            probs[bridge] = 0.30
            probs[target] = 0.18
            probs[decoy] = 0.12
            if order_ready:
                probs[target] = 0.82
                probs[bridge] = 0.08
                probs[decoy] = 0.04
            elif carryover > 0:
                probs[target] = 0.74
                probs[bridge] = 0.10
                probs[decoy] = 0.06
            elif wrong_order:
                probs[decoy] = 0.44
                probs[target] = 0.08
                probs[bridge] = 0.20
            probs = probs / probs.sum()
            action = int(rng.choice(N_ACTIONS, p=probs))

            signal = -1.0
            if action == bridge:
                signal += 0.20
            if action == target:
                signal += 0.60
            if order_ready and action == target:
                signal += 4.4
            if carryover > 0 and action == target:
                signal += 2.2
            if wrong_order and action == target:
                signal -= 2.4
            if wrong_order and action == decoy:
                signal += 0.60
            signal += rng.normal(0.0, 0.10 if action == target else 0.60)
            buy = signal > 1.0

            if buy:
                if order_ready and action == target:
                    reward = float(rng.integers(12, 18) * 100)
                    if rng.random() < 0.35:
                        reward += float(rng.integers(8, 13) * 100)
                elif carryover > 0 and action == target:
                    reward = float(rng.integers(7, 11) * 100)
                elif action == target:
                    reward = float(rng.integers(2, 5) * 100)
                else:
                    reward = 100.0
            else:
                reward = 0.0

            next_page = int((page + rng.choice([0, 1, 2], p=[0.42, 0.38, 0.20])) % N_PAGES)
            next_regime = regime
            next_carryover = max(0, carryover - 1)
            if order_ready and action == target:
                next_regime = (regime + 1) % 3
                next_carryover = 3

            pre_session = []
            for shift in range(LAG_1):
                snapshot = latent + 0.01 * shift + rng.normal(0.0, 0.01, FEATURE_DIM)
                snapshot[3] = float(user_type) / 2.0
                snapshot[4] = float(bridge_idx)
                snapshot[5] = float(target_idx)
                snapshot[6] = float(decoy_idx)
                snapshot[7] = float(min(carryover, 3)) / 3.0
                snapshot[8] = float(regime) / 2.0
                pre_session.append(snapshot)

            # Keep within-session observations weak so summation-based methods
            # cannot easily recover the user mapping or trigger sequence.
            state_vec = rng.normal(0.0, 0.003, FEATURE_DIM)
            next_vec = rng.normal(0.0, 0.003, FEATURE_DIM)

            within_pre = list(within_history[-9:]) + [state_vec]
            within_next = list(within_history[-9:]) + [state_vec, next_vec]
            features.append(
                [
                    torch.tensor(np.asarray(pre_session), dtype=torch.float32),
                    torch.tensor(np.asarray(within_pre), dtype=torch.float32),
                    torch.tensor(np.asarray(within_next), dtype=torch.float32),
                    int(action),
                    float(reward),
                    int(page),
                    int(next_page),
                    int(user_id),
                ]
            )

            within_history.append(state_vec)
            prev_token = token
            prev_action = action
            regime = next_regime
            carryover = next_carryover
            latent = 0.985 * latent + 0.015 * regime_vectors[next_regime]
            page = next_page

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-users", type=int, default=400)
    parser.add_argument("--test-users", type=int, default=120)
    parser.add_argument("--min-steps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["default", "brqn_favor", "brqn_linear_sparse", "brqn_mechanism", "brqn_order_uncertainty", "brqn_horizon_support", "brqn_presession_linear", "brqn_regime_action", "brqn_user_mapping", "brqn_user_mapping_horizon", "brqn_user_mapping_longbonus", "brqn_simple"], default="default")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "brqn_favor":
        generator = generate_split_brqn_favor
    elif args.mode == "brqn_linear_sparse":
        generator = generate_split_brqn_linear_sparse
    elif args.mode == "brqn_mechanism":
        generator = generate_split_brqn_mechanism
    elif args.mode == "brqn_order_uncertainty":
        generator = generate_split_brqn_order_uncertainty
    elif args.mode == "brqn_horizon_support":
        generator = generate_split_brqn_horizon_support
    elif args.mode == "brqn_presession_linear":
        generator = generate_split_brqn_presession_linear
    elif args.mode == "brqn_regime_action":
        generator = generate_split_brqn_regime_action
    elif args.mode == "brqn_user_mapping":
        generator = generate_split_brqn_user_mapping
    elif args.mode == "brqn_user_mapping_horizon":
        generator = generate_split_brqn_user_mapping_horizon
    elif args.mode == "brqn_user_mapping_longbonus":
        generator = generate_split_brqn_user_mapping_longbonus
    elif args.mode == "brqn_simple":
        generator = generate_split_brqn_simple
    else:
        generator = generate_split_default

    train = generator(args.train_users, args.min_steps, args.max_steps, args.seed)
    test = generator(args.test_users, args.min_steps, args.max_steps, args.seed + 1)

    (outdir / "training_feature_batch.pkl").write_bytes(pickle.dumps(train))
    test_bytes = pickle.dumps(test)
    (outdir / "testing_feature_batch.pkl").write_bytes(test_bytes)
    (outdir / "testing.pkl").write_bytes(test_bytes)

    print(f"train samples: {len(train)}")
    print(f"test samples: {len(test)}")
    print(f"mode: {args.mode}")
    print(f"wrote: {outdir / 'training_feature_batch.pkl'}")
    print(f"wrote: {outdir / 'testing_feature_batch.pkl'}")
    print(f"wrote: {outdir / 'testing.pkl'}")


if __name__ == "__main__":
    main()
