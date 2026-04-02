import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange

CARD_TYPE_MAP = {
    'O': 0,  # Other
    'L': 1,  # Land
    'R': 2,  # Ramp
    'B': 3,  # Bomb
    'D': 4   # Draw
}

@njit(inline='always')
def find_bottom_card_idx(hand, priority=(0, 3, 2, 1, 4)):
    """
    Select the index of the card that should be placed on the bottom of the deck
    during an extra mulligan.
    
    Cards are prioritized according to a fixed hierarchy, from highest to lowest
    priority for bottoming. Default priority:
        Other (0) > Bomb (3) > Ramp (2) > Land (1) > Draw (4)
    
    The function scans the first 7 cards of the hand (the hand size during an
    extra mulligan) and returns the index of the first card matching the highest
    priority found.
    
    Parameters
    ----------
    hand : np.ndarray
        Array of card types representing the current hand.
    priority : tuple of int, optional
        Card priority order for bottoming, default is (0, 3, 2, 1, 4).
    
    Returns
    -------
    int
        Index of the selected card to put on the bottom of the deck.
    """

    for selected_card in priority: #choose one card type
        for idx in range(len(hand)): #loop through hand to find it
            if hand[idx] == selected_card: #if found, return it. Otherwise, go to next type
                return idx

    return 0 #fallback for numba, does not happen by design
    
@njit(inline='always')
def mulligan(base_deck, t_land, t_ramp, t_bomb, t_draw, N_mulligans, priority=(0, 3, 2, 1, 4)):
    """
    Perform a mulligan procedure on a deck to attempt to satisfy minimum card
    requirements for the opening hand.
    
    The function simulates drawing a hand of 7 cards (or fewer if extra mulligans
    are applied), optionally bottoming lower-priority cards, and checks if the
    hand satisfies minimum requirements for Land, Ramp, Bomb, and Draw cards.
    
    Parameters
    ----------
    base_deck : np.ndarray
        The initial deck array encoding card types.
    t_land : int
        Minimum number of Land cards required in the opening hand.
    t_ramp : int
        Minimum number of Ramp cards required in the opening hand.
    t_bomb : int
        Minimum number of Bomb cards required in the opening hand.
    t_draw : int
        Minimum number of Draw cards required in the opening hand.
    N_mulligans : int
        Maximum number of mulligans allowed.
    priority : tuple of int, optional
        Card priority order for bottoming during extra mulligans, default is (0,3,2,1,4).
    
    Returns
    -------
    deck : np.ndarray
        Deck array after shuffling and applying any bottoming.
    hand_size : int
        Number of cards in the opening hand after mulligans.
    attempt : int
        Number of mulligan attempts performed to reach a valid hand.
    """
    
    deck = np.empty_like(base_deck)

    min_cards = t_land + t_ramp + t_bomb + t_draw

    for attempt in range(N_mulligans + 1):

        # --- Determine hand size ---
        if attempt <= 1:
            hand_size = 7
            n_bottom = 0
        else:
            n_bottom = attempt - 1
            hand_size = 7 - n_bottom

        # --- Stop if impossible to satisfy requirements ---
        if hand_size < min_cards:
            return deck, hand_size + 1, attempt - 1

        # --- Shuffle fresh deck ---
        deck[:] = base_deck
        np.random.shuffle(deck)

        # --- Apply bottoming (if needed) ---
        if n_bottom > 0:

            # Work on first 7 cards only
            hand = deck[:7]

            for i in range(n_bottom):
                current_size = 7 - i

                # select card to bottom
                idx = find_bottom_card_idx(hand, priority)
                bottom_card = hand[idx]

                # shift left
                for j in range(idx, current_size - 1):
                    hand[j] = hand[j + 1]

                # rebuild deck in-place
                new_hand_size = current_size - 1

                # front: updated hand
                deck[:new_hand_size] = hand[:new_hand_size]

                # middle: rest of deck
                tail_size = deck.size - 7
                deck[new_hand_size:new_hand_size + tail_size] = deck[7:7 + tail_size]

                # bottom card
                deck[-1] = bottom_card

                # update hand view
                hand = deck[:new_hand_size]

        # --- Evaluate hand ---
        hand = deck[:hand_size]
        n_land = 0
        n_ramp = 0
        n_bomb = 0
        n_draw = 0

        # manual counting (faster than np.sum in Numba loops)
        for k in range(hand_size):
            c = hand[k]
            if c == 1:
                n_land += 1
            elif c == 2:
                n_ramp += 1
            elif c == 3:
                n_bomb += 1
            elif c == 4:
                n_draw += 1

        if (n_land >= t_land and
            n_ramp >= t_ramp and
            n_bomb >= t_bomb and
            n_draw >= t_draw):
            return deck, hand_size, attempt

    # fallback: last attempt
    return deck, hand_size, attempt

@njit(parallel=True)
def simulate_games(N_sim, base_deck, t_land, t_ramp, t_bomb, t_draw, gameplan, 
                   N_mulligans, priority =(0,3,2,1,4)):
    """
    Run a Monte Carlo simulation of multiple games, evaluating success of a
    predefined per-turn gameplan.
    
    Each simulation applies the mulligan procedure, draws cards per turn, and
    attempts to play required card types according to the gameplan. Optional Draw
    cards are resolved by drawing additional cards from the deck. Failures are
    tracked by turn and operation.
    
    Parameters
    ----------
    N_sim : int
        Number of simulated games.
    base_deck : np.ndarray
        Deck template encoding card types (0=Other, 1=Land, 2=Ramp, 3=Bomb, 4=Draw).
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
    N_mulligans : int
        Maximum number of mulligans allowed.
    priority : tuple of int, optional
        Card priority order for bottoming during extra mulligans, default is (0,3,2,1,4).
    
    Returns
    -------
    success_rate : float
        Percentage of games that successfully complete the gameplan.
    fail_summary : np.ndarray
        2D array counting failures by turn and operation index.
    mulligan_stats : np.ndarray
        Array recording the number of mulligans performed in each game.
    """  
    
    # --- Arrays to record simulation results ---
    successes = np.zeros(N_sim, dtype=np.bool_) #1 indicates success, 0 indicates failure in each game
    fail_turns = np.full(N_sim, -1, dtype=np.int8) #if failure, indicates the turn in which the plan failed
    fail_ops = np.full(N_sim, -1, dtype=np.int8) #if failed, indicates the operation in which the plan failed
    mulligan_stats = np.full(N_sim, -1, dtype=np.int64) #number of mulligans performed at each game

    # --- Game simulation (parallel) ---
    for i in prange(N_sim):
        deck, hand_size, N_mulligan_performed = mulligan(base_deck, t_land, t_ramp, t_bomb, t_draw, N_mulligans, priority)
        mulligan_stats[i] = N_mulligan_performed #Accumulate statistics about mulligan number

        #Draw all cards that are seen until T5 (with an extra card if we have T1 draw spell
        hand = np.empty(12, dtype=np.uint8)
        hand[:hand_size] = deck[:hand_size]
        deck_pos = hand_size

        #Default values
        success = True
        fail_turn = -1
        fail_op = -1
        fail_card_type = -1
        
        for turn_i in range(len(gameplan)): #Turn sequence
            
            # Draw a card at start of turn
            hand[hand_size] = deck[deck_pos]
            hand_size += 1
            deck_pos += 1
               
            # Play operations in gameplan
            for operation_idx in range(len(gameplan[turn_i])):
                card_type = gameplan[turn_i][operation_idx]
                
                # --- Optional Draw card at T1 ---
                if card_type == 4 and turn_i == 0: #turn 1 and draw spell in hand
                    found_draw_T1 = False
                    for j in range(hand_size): #loop through hand... 
                        if hand[j] == 4: # ...to search for draw spell
                            # Cast draw spell (remove it from hand)
                            hand[j:hand_size-1] = hand[j+1:hand_size]
                            hand_size -= 1
                            found_draw_T1 = True
                            # Resolve effect: Draw next card from deck
                            if deck_pos < len(deck):
                                hand[hand_size] = deck[deck_pos]
                                hand_size += 1
                                deck_pos += 1
                            break
                    # Optional: skip if Draw not in hand in T1
                    continue
                
                # --- Required card play ---
                found = False
                for j in range(hand_size): #loop through hand...
                    if hand[j] == card_type: #... to play the required card of type card_type
                        hand[j:hand_size-1] = hand[j+1:hand_size]
                        hand_size -= 1
                        found = True
                        break
                
                if not found: #looked for required card and failed -> Gamplan failed
                    success = False
                    fail_turn = turn_i
                    fail_op = operation_idx
                    fail_card_type = card_type
                    break  # stop this game
            
            if not success:
                break

        # Save game results for later analysis
        successes[i] = success
        fail_turns[i] = fail_turn
        fail_ops[i] = fail_op
    
    # --- Aggregate results ---
    success_rate = np.mean(successes) * 100
    fail_summary = np.zeros((len(gameplan), 2), dtype=np.int64)  # assuming max 2 operations per turn
    
    for i in range(N_sim):
        if not successes[i]:
            t = fail_turns[i]
            o = fail_ops[i]
            if 0 <= t < len(gameplan) and 0 <= o < 2:
                fail_summary[t, o] += 1
    
    return success_rate, fail_summary, mulligan_stats

def print_simulation_report(result, fail_summary, mulligan_stats, N_sim, N_mulligans):
    """
    Print a formatted report of simulation results, including success rate,
    failure breakdown, and mulligan usage statistics.
    
    Parameters
    ----------
    result : float
        Overall success rate (percentage).
    fail_summary : np.ndarray
        2D array of failure counts by turn and operation.
    mulligan_stats : np.ndarray
        Array recording the number of mulligans performed in each game.
    N_sim : int
        Total number of simulated games.
    N_mulligans : int
        Maximum number of mulligans allowed.
    """
    
    success_rate = result
    fail_rate = 100 - success_rate
    total_failures = np.sum(fail_summary)

    print("\n✅ Success rate: {:.2f}%".format(success_rate))
    print("❌ Failure rate: {:.2f}%".format(fail_rate))

    # --- Mulligan usage breakdown ---
    print("\n🎲 Mulligan usage breakdown:")
    for m in range(N_mulligans + 1):
        count = np.sum(mulligan_stats == m)
        pct = count / N_sim * 100
        print(f"{m} mulligan(s): {count} games → {pct:.2f}%")

    # --- Failure breakdown ---
    operation_labels = [["Play Land"], ["Play Land", "Play Ramp"], ["Play Land"], ["Play Land", "Play Bomb"]]

    print("\n🔎 Failure breakdown by operation (as % of ALL games):")
    for turn in range(len(fail_summary)):
        for op in range(len(operation_labels[turn])):
            print(f" Turn {turn+1} - {operation_labels[turn][op]}: {fail_summary[turn, op] / N_sim * 100:6.3f}%")

    print("\n📊 Conditional breakdown (as % of FAILED games):")
    for turn in range(len(fail_summary)):
        for op in range(len(operation_labels[turn])):
            pct = 0.0
            if total_failures > 0:
                pct = fail_summary[turn, op] / total_failures * 100
            print(f" Turn {turn+1} - {operation_labels[turn][op]}: {pct:6.2f}%")