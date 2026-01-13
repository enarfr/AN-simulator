import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange

@njit(inline='always')
def find_bottom_card_idx(hand):
    """
    Select the index of the card that should be placed on the bottom of the deck
    during an extra mulligan.
    
    Cards are prioritized according to a fixed hierarchy, from highest to lowest
    priority for bottoming:
        Other (0) > Bomb (3) > Ramp (2) > Land (1) > Draw (4)
    
    The function scans the first 7 cards of the hand (the hand size during an
    extra mulligan) and returns the index of the first card matching the highest
    priority found.
    
    Parameters
    ----------
    hand : np.ndarray
        Array of card types representing the current hand.
    
    Returns
    -------
    int
        Index of the selected card to put on the bottom of the deck.
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
    Perform the mulligan procedure with optional priority-based extra mulligan.
    
    The function attempts up to three opening hands:
    1. First attempt: draw 7 cards.
    2. Second attempt: draw 7 cards again if thresholds are not met.
    3. Optional extra mulligan: draw 7 cards, then bottom one card according to
       a fixed priority rule, keeping a 6-card hand.
    
    The mulligan succeeds early if the hand meets the minimum required counts of
    Land, Ramp, Bomb, and Draw cards.
    
    Parameters
    ----------
    base_deck : np.ndarray
        Deck template containing encoded card types.
    t_land : int
        Minimum number of Land cards required in the opening hand.
    t_ramp : int
        Minimum number of Ramp cards required in the opening hand.
    t_bomb : int
        Minimum number of Bomb cards required in the opening hand.
    t_draw : int
        Minimum number of Draw cards required in the opening hand.
    extra_mulligan : bool
        Whether to allow the priority-based extra mulligan.
    
    Returns
    -------
    deck : np.ndarray
        Shuffled deck after mulligan adjustments.
    hand_size : int
        Number of cards kept in the opening hand (6 or 7).
    mulligan_count : int
        Number of mulligans used (0, 1, or 2).
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
            return deck, 7, attempt  # 0 or 1 mulligan used

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

        return deck, 6, 2  # extra mulligan used

    # --- Otherwise, return 7-card hand from last shuffle ---
    return deck, 7, 1  # failed on second attempt, no extra mulligan

@njit(parallel=True)
def simulate_games(N_sim, base_deck, t_land, t_ramp, t_bomb, t_draw, gameplan, extra_mulligan):
    """
    Run a detailed Monte Carlo simulation of N_sim games in parallel, tracking
    success rates, mulligan usage, and failure causes.
    
    Each simulation:
    - Applies the mulligan procedure with optional extra mulligan.
    - Draws one card at the start of each turn.
    - Attempts to execute a predefined per-turn gameplan by playing required
      card types from hand.
    - Optionally resolves Draw cards by drawing an additional card from the deck.
    
    A game is marked as a failure if a required card cannot be played at the
    specified turn and operation.
    
    Parameters
    ----------
    N_sim : int
        Number of simulated games.
    base_deck : np.ndarray
        Deck template encoding card types:
        0=Other, 1=Land, 2=Ramp, 3=Bomb, 4=Draw.
    t_land : int
        Minimum number of Land cards required in the opening hand.
    t_ramp : int
        Minimum number of Ramp cards required in the opening hand.
    t_bomb : int
        Minimum number of Bomb cards required in the opening hand.
    t_draw : int
        Minimum number of Draw cards required in the opening hand.
    gameplan : list of np.ndarray
        Per-turn list of required card types to be played.
    extra_mulligan : bool
        Whether to enable the priority-based extra mulligan.
    
    Returns
    -------
    success_rate : float
        Percentage of games that successfully complete the gameplan.
    fail_summary : np.ndarray
        2D array counting failures by turn and operation index.
    mulligan_stats : np.ndarray
        Array counting how many games used 0, 1, or 2 mulligans.
    """
    
    # --- Arrays to record simulation results ---
    successes = np.zeros(N_sim, dtype=np.bool_)
    fail_turns = np.full(N_sim, -1, dtype=np.int8)
    fail_ops = np.full(N_sim, -1, dtype=np.int8)
    fail_cards = np.full(N_sim, -1, dtype=np.int8)
    mulligan_stats = np.zeros(3, dtype=np.int64)  # index = number of mulligans used

    
    # --- Main parallel simulation ---
    for i in prange(N_sim):
        deck, hand_size, N_mulligan = draw_with_mulligan_priority(base_deck, t_land, t_ramp, t_bomb, t_draw, extra_mulligan)
        mulligan_stats[N_mulligan] += 1 #Accumulate statistics about mulligan number
        
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
    
    return success_rate, fail_summary, mulligan_stats

def print_simulation_report(result, fail_summary, mulligan_stats, N_sim):
    """
    Print a formatted summary of simulation results, including success rate,
    failure breakdown, and mulligan usage statistics.
    
    The report includes:
    - Overall success and failure rates.
    - Distribution of mulligan usage across simulations.
    - Failure rates by turn and operation, expressed both as a percentage of all
      games and conditionally as a percentage of failed games.
    
    Parameters
    ----------
    result : float
        Overall success rate (percentage).
    fail_summary : np.ndarray
        2D array of failure counts by turn and operation.
    mulligan_stats : np.ndarray
        Array counting how many games used 0, 1, or 2 mulligans.
    N_sim : int
        Total number of simulated games.
    """

    success_rate = result
    fail_rate = 100 - success_rate
    total_failures = np.sum(fail_summary)

    print("\n✅ Success rate: {:.2f}%".format(success_rate))
    print("❌ Failure rate: {:.2f}%".format(fail_rate))

    # Print mulligan info
    total_mulliganed = np.sum(mulligan_stats)
    print("\n🎲 Mulligan usage breakdown:")
    for i in range(3):
        pct = mulligan_stats[i] / total_mulliganed * 100 if total_mulliganed > 0 else 0.0
        print("{} mulligan(s): {} games → {:.2f}%".format(i, mulligan_stats[i], pct))

    # Existing failure breakdown
    operation_labels = [ ["Play Land"], ["Play Land", "Play Ramp"], ["Play Land"], ["Play Land", "Play Bomb"] ]

    print("\n🔎 Failure breakdown by operation (as % of ALL games):")
    for turn in range(4):
        for op in range(len(operation_labels[turn])):
            print(" Turn {} - {}: {:6.3f}%".format(turn + 1, operation_labels[turn][op], fail_summary[turn, op] / N_sim * 100))


    print("\n📊 Conditional breakdown (as % of FAILED games):")
    for turn in range(4):
        for op in range(len(operation_labels[turn])):
            pct = 0.0
            if total_failures > 0:
                pct = fail_summary[turn, op] / total_failures * 100
            print(" Turn {} - {}: {:6.2f}%".format(turn+1, operation_labels[turn][op], pct))