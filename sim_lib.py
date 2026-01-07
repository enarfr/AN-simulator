import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange

@njit(inline='always')
def find_bottom_card_idx(hand):
    """
    Return the index of the card to put on bottom of the deck.
    Type Priority: Other (0) > Bomb (3) > Ramp (2) > Draw (4) > Land (1)
    """
    for priority_card in (0, 3, 2, 1, 4): #Draw
        for idx in range(7): #in extra mulligan my hand is always 7 cards
            if hand[idx] == priority_card:
                return idx
    return 0  # fallback (should never happen

# --- Mulligan helper ---
@njit(inline='always')
def draw_with_mulligan_priority(base_deck, t_land, t_ramp, t_bomb, t_draw, extra_mulligan):
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
        n_draw = np.sum(hand == 4)


        if n_land >= t_land and n_ramp >= t_ramp and n_bomb >= t_bomb and n_draw >= t_draw:
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

from numba import njit, prange
import numpy as np

@njit(parallel=True)
def simulate_games_detailed(N_sim, base_deck, t_land, t_ramp, t_bomb, t_draw, gameplan, extra_mulligan):
    """
    Simulate N_sim games in parallel with optional Draw cards.
    
    Parameters
    ----------
    N_sim : int
        Number of simulations to run.
    base_deck : np.ndarray
        Deck template (uint8), including card types: 0=Other,1=Land,2=Ramp,3=Bomb,4=Draw
    t_land, t_ramp, t_bomb : int
        Minimum number of each card type to keep initial hand
    gameplan : list of np.ndarray
        List of per-turn operations (required card types to play)
    extra_mulligan : bool
        Whether to apply extra mulligan logic
    t_draw : int, optional
        Minimum Draw cards to keep initial hand
    
    Returns
    -------
    success_rate : float
        Percentage of successful games
    fail_summary : np.ndarray
        2D array of failure counts by turn and operation
    """
    
    # --- Arrays to record simulation results ---
    successes = np.zeros(N_sim, dtype=np.bool_)
    fail_turns = np.full(N_sim, -1, dtype=np.int8)
    fail_ops = np.full(N_sim, -1, dtype=np.int8)
    fail_cards = np.full(N_sim, -1, dtype=np.int8)
    
    # --- Main parallel simulation ---
    for i in prange(N_sim):
        deck, hand_size = draw_with_mulligan_priority(base_deck, t_land, t_ramp, t_bomb, t_draw, extra_mulligan)
        hand = np.empty(12, dtype=np.uint8)  # increased size for optional draw
        hand[:hand_size] = deck[:hand_size]
        deck_pos = hand_size
        
        success = True
        fail_turn = -1
        fail_op = -1
        fail_card = -1
        
        for turn_i in range(len(gameplan)):
            # Draw a card at start of turn
            if deck_pos < len(deck):
                hand[hand_size] = deck[deck_pos]
                hand_size += 1
                deck_pos += 1
            
            # Play operations in gameplan
            for operation_idx in range(len(gameplan[turn_i])):
                card_type = gameplan[turn_i][operation_idx]
                
                # --- Optional Draw card at T1 ---
                if card_type == 4:
                    found_draw = False
                    for j in range(hand_size):
                        if hand[j] == 4:
                            # Remove Draw card from hand
                            hand[j:hand_size-1] = hand[j+1:hand_size]
                            hand_size -= 1
                            found_draw = True
                            # Draw next card from deck
                            if deck_pos < len(deck):
                                hand[hand_size] = deck[deck_pos]
                                hand_size += 1
                                deck_pos += 1
                            break
                    # Optional: skip if Draw not in hand
                    continue
                
                # --- Required card play ---
                found = False
                for j in range(hand_size):
                    if hand[j] == card_type:
                        hand[j:hand_size-1] = hand[j+1:hand_size]
                        hand_size -= 1
                        found = True
                        break
                
                if not found:
                    success = False
                    fail_turn = turn_i
                    fail_op = operation_idx
                    fail_card = card_type
                    break  # stop this game
            
            if not success:
                break
        
        successes[i] = success
        fail_turns[i] = fail_turn
        fail_ops[i] = fail_op
        fail_cards[i] = fail_card
    
    # --- Aggregate results ---
    success_rate = np.mean(successes) * 100
    fail_summary = np.zeros((len(gameplan), 2), dtype=np.int64)  # assuming max 2 ops per turn
    
    for i in range(N_sim):
        if not successes[i]:
            t = fail_turns[i]
            o = fail_ops[i]
            if 0 <= t < len(gameplan) and 0 <= o < 2:
                fail_summary[t, o] += 1
    
    return success_rate, fail_summary


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


