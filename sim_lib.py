import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange

@njit(inline='always')
def find_bottom_card_idx(hand):
    """
    Return the index of the card to put on bottom of the deck.
    Priority: Other (0) > Bomb (3) > Ramp (2) > Land (1)
    """
    for priority_card in (0, 3, 2, 1):
        for idx in range(7): #in extra mulligan my hand is always 7 cards
            if hand[idx] == priority_card:
                return idx
    return 0  # fallback (should never happen)


# --- Mulligan helper ---
@njit(inline='always')
def draw_with_mulligan_priority(base_deck, t_land, t_ramp, t_bomb, extra_mulligan):
    """
    Mulligan procedure:
      1. Try 7 cards → keep if criteria met.
      2. Try again (free mulligan) → keep if criteria met.
      3. If both fail and extra_mulligan=True:
         - Draw 7 cards.
         - Put one card on bottom of deck by priority:
             Other (0) > Bomb (3) > Ramp (2) > Land (1)
         - Keep remaining 6 cards.
      4. If extra_mulligan=False → just keep last 7.
    """
    deck = np.empty_like(base_deck)

    # --- Attempt 1 and 2: 7-card hands ---
    for attempt in range(2):
        deck[:] = base_deck
        np.random.shuffle(deck)
        hand = deck[:7]

        n_land = np.sum(hand == 1)
        n_ramp = np.sum(hand == 2)
        n_bomb = np.sum(hand == 3)

        if n_land >= t_land and n_ramp >= t_ramp and n_bomb >= t_bomb:
            return deck, 7  # keep 7-card hand

    # --- Attempt 3 (only if extra_mulligan is True) ---
    if extra_mulligan:
        deck[:] = base_deck
        np.random.shuffle(deck)
        hand = deck[:7]

        bottom_idx = find_bottom_card_idx(hand)
        bottom_card = hand[bottom_idx]

        # Remove chosen card from hand by shifting left
        hand[bottom_idx:6] = hand[bottom_idx + 1:7]

        # Build new deck: 6 cards from hand + the rest of the deck + bottomed card
        temp = deck[7:].copy()
        deck[:-1] = np.concatenate((hand[:6], temp))
        deck[-1] = bottom_card

        return deck, 6

    # --- Otherwise, return 7-card hand from last shuffle ---
    return deck, 7


# --- Main simulation with detailed failure reporting ---
@njit(parallel=True)
def simulate_games_detailed(N_sim, base_deck, t_land, t_ramp, t_bomb, gameplan, extra_mulligan):
    """
    Simulate N_sim games in parallel.
    For failures, record:
        - turn index (0–3)
        - operation index within turn (e.g. 0=land, 1=ramp, etc.)
        - card type required at failure
    """
    successes = np.zeros(N_sim, dtype=np.bool_)
    fail_turns = np.full(N_sim, -1, dtype=np.int8)   # which turn failed
    fail_ops = np.full(N_sim, -1, dtype=np.int8)     # which operation failed
    fail_cards = np.full(N_sim, -1, dtype=np.int8)   # what card type caused failure

    for i in prange(N_sim):
        deck, hand_size = draw_with_mulligan_priority(base_deck, t_land, t_ramp, t_bomb, extra_mulligan)
        hand = np.empty(11, dtype=np.uint8)
        hand[:hand_size] = deck[:hand_size]
        deck_pos = hand_size

        success = True
        fail_turn = -1
        fail_op = -1
        fail_card = 0

        for turn_i in range(4):
            # Draw a card
            hand[hand_size] = deck[deck_pos]
            hand_size += 1
            deck_pos += 1

            # Play required cards in this turn
            for operation_idx in range(len(gameplan[turn_i])): 
                card_type = gameplan[turn_i][operation_idx] #required card type to play, according to gameplan
                found = False #initial state

                for j in range(hand_size): #we check in our hand if we have such card type
                    if hand[j] == card_type:
                        hand[j:hand_size - 1] = hand[j + 1:hand_size] # Found it! -> Remove the card by shifting left 
                        hand_size -= 1
                        found = True
                        break

                if not found:
                    success = False
                    fail_turn = turn_i
                    fail_op = op_i
                    fail_card = card_type
                    break  # stop this game
                    
            if not success:
                break

        successes[i] = success
        fail_turns[i] = fail_turn
        fail_ops[i] = fail_op
        fail_cards[i] = fail_card

    # --- Aggregate results ---
    success_rate = np.mean(successes)

    # 4 turns × up to 2 operations per turn (worst-case)
    fail_summary = np.zeros((4, 2), dtype=np.int64)

    for i in range(N_sim):
        if not successes[i]:
            t = fail_turns[i]
            o = fail_ops[i]
            if 0 <= t < 4 and 0 <= o < 2:
                fail_summary[t, o] += 1

    total_failures = N_sim - np.sum(successes)

    return success_rate * 100, fail_summary, total_failures

def print_simulation_report(result, fail_summary, N_sim):
    """
    Pretty-print simulation results and failure breakdown.
    """

    # --- Compute basic metrics ---
    success_rate = result
    fail_rate = 100 - success_rate
    total_failures = np.sum(fail_summary)

    # --- Operation labels (by turn) ---
    operation_labels = [
        ["Play Land"],                 # Turn 1
        ["Play Land", "Play Ramp"],    # Turn 2
        ["Play Land"],                 # Turn 3
        ["Play Land", "Play Bomb"]     # Turn 4
    ]

    # --- Present readable output ---
    print(f"\n✅ Success rate: {success_rate:.2f}%")
    print(f"❌ Failure rate: {fail_rate:.2f}%")

    print("\n🔎 Failure breakdown by operation (as % of ALL games):")
    for turn in range(4):
        for op in range(len(operation_labels[turn])):
            print(f" Turn {turn+1} - {operation_labels[turn][op]}:"
                  f" {fail_summary[turn, op] / N_sim * 100:6.3f}%")

    print("\n📊 Conditional breakdown (as % of FAILED games):")
    for turn in range(4):
        for op in range(len(operation_labels[turn])):
            pct = 0.0
            if total_failures > 0:
                pct = fail_summary[turn, op] / total_failures * 100
            print(f" Turn {turn+1} - {operation_labels[turn][op]}: {pct:6.2f}%")


