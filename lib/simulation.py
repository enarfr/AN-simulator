import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit, prange

CARD_TYPE_MAP = {
    'O': 0,  # Other
    'L': 1,  # Land
    'R': 2,  # Ramp
    'B': 3,  # Bomb
    'D': 4   # Draw
}

@njit(inline='always')
def shuffle_deck(deck):
    """
    Perform an in-place Fisher-Yates shuffle on a deck array.

    Preferred over np.random.shuffle inside Numba parallel regions because it
    operates entirely within the Numba runtime, using Numba's thread-local PRNG
    state. This guarantees correctness and avoids any Python-side RNG contention
    when called from prange workers.

    Parameters
    ----------
    deck : np.ndarray
        1-D array of card type codes to be shuffled in place.

    Returns
    -------
    None
        The array is modified in place; nothing is returned.
    """
 
    n = len(deck)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        deck[i], deck[j] = deck[j], deck[i]
    
#Mulligan
@njit(inline='always')
def mulligan(base_deck, t_land, t_ramp, t_bomb, t_draw, N_mulligans, priority=(0, 3, 2, 1, 4)):
    """
    Perform a London Mulligan procedure to attempt to satisfy minimum card
    requirements for the opening hand.

    Each attempt draws 7 cards fresh from a reshuffled deck. The first mulligan
    is free (no bottoming). From the second mulligan onward, n_bottom = attempt - 1
    cards must be placed on the bottom of the deck before keeping the hand.

    Bottoming respects a two-phase rule:
      1. Required cards (up to t_land Lands, t_ramp Ramp, t_bomb Bombs, t_draw Draw)
         are locked in and protected from being bottomed.
      2. The bottoming priority is applied only to the remaining free cards.

    Example
    -------
    t_land=2, t_ramp=1, t_draw=0, t_bomb=0, attempt=3 (n_bottom=2), hand=(L,L,R,O,O,R,O),
    priority=(O>R>B>L>D):
      - Lock L, L, R as required.
      - Free pool: (O, O, R, O). Bottom 2 by priority → bottom O, O.
      - Final hand: (L, L, R, R, O). Bottom of deck: (..., O, O).

    Parameters
    ----------
    base_deck : np.ndarray
        The initial deck array encoding card types
        (0=Other, 1=Land, 2=Ramp, 3=Bomb, 4=Draw).
    t_land : int
        Minimum number of Land cards required in the opening hand.
    t_ramp : int
        Minimum number of Ramp cards required in the opening hand.
    t_bomb : int
        Minimum number of Bomb cards required in the opening hand.
    t_draw : int
        Minimum number of Draw cards required in the opening hand.
    N_mulligans : int
        Maximum number of mulligans allowed (0 = no mulligans, keep first hand).
    priority : tuple of int, optional
        Card type order for bottoming, from most to least preferred to bottom.
        Only applied to cards not locked in as required. Default: (0,3,2,1,4).

    Returns
    -------
    deck : np.ndarray
        Deck array after shuffling and bottoming. deck[:hand_size] is the
        opening hand; deck[-n_bottom:] are the bottomed cards (if any).
    hand_size : int
        Number of cards in the opening hand.
    attempt : int
        Number of mulligan attempts performed (0 = kept first hand).
    """

    deck = np.empty_like(base_deck)
    min_cards = t_land + t_ramp + t_bomb + t_draw
    
    # Buffer for bottom card values — at most N_mulligans cards are ever bottomed
    bottom_cards = np.empty(N_mulligans, dtype=np.uint8)
    
    for attempt in range(N_mulligans + 1):

        # --- Determine hand size and number of cards to bottom ---
        if attempt <= 1:
            # attempt=0: first draw, no bottoming
            # attempt=1: first (free) mulligan, still no bottoming
            hand_size = 7
            n_bottom = 0
        else:
            # attempt=2 → bottom 1, keep 6
            # attempt=3 → bottom 2, keep 5  ... etc.
            n_bottom = attempt - 1
            hand_size = 7 - n_bottom

        # --- Stop if the hand would be too small to satisfy requirements ---
        if hand_size < min_cards:
            # Return the last valid deck state; hand_size+1 and attempt-1
            # reflect the previous (last viable) attempt.
            return deck, hand_size + 1, attempt - 1

        # --- Shuffle a fresh copy of the deck ---
        deck[:] = base_deck
        shuffle_deck(deck)

        # --- Apply bottoming if required ---
        if n_bottom > 0:

            # Save the 7-card draw before any in-place modifications
            hand7 = deck[:7].copy()

            # Phase 1: lock in required cards (protect them from being bottomed).
            # Walk the hand in order, satisfying each requirement with the first
            # matching card found. Track which indices are locked.
            is_required = np.zeros(7, dtype=np.bool_) #All false values
            need_land = t_land
            need_ramp = t_ramp
            need_bomb = t_bomb
            need_draw = t_draw

            for k in range(7):
                c = hand7[k]
                if c == 1 and need_land > 0:
                    is_required[k] = True
                    need_land -= 1
                elif c == 2 and need_ramp > 0:
                    is_required[k] = True
                    need_ramp -= 1
                elif c == 3 and need_bomb > 0:
                    is_required[k] = True
                    need_bomb -= 1
                elif c == 4 and need_draw > 0:
                    is_required[k] = True
                    need_draw -= 1

            # Phase 2: select n_bottom cards to bottom from the free pool only.
            # Walk the priority tuple; for each priority type, scan the hand for
            # the first free (non-required, not-yet-bottomed) matching card.
            # Track by index to handle duplicate card types correctly.
            
            will_bottom = np.zeros(7, dtype=np.bool_)
            n_selected = 0
            
            for pcard in priority:
                for k in range(7):
                    if not is_required[k] and not will_bottom[k] and hand7[k] == pcard:
                        will_bottom[k] = True
                        bottom_cards[n_selected] = hand7[k]
                        n_selected += 1
                        if n_selected == n_bottom:
                            break   # inner break — done selecting, exit inner loop
                
                if n_selected == n_bottom:
                    break           # outer break — also exit outer loop

            # Safety fallback: if the free pool was exhausted before n_bottom cards
            # were selected (shouldn't happen with valid deck/requirements), fill
            # remaining slots from any not-yet-selected card.
            if n_selected < n_bottom:
                for k in range(7):
                    if not is_required[k] and not will_bottom[k]:
                        will_bottom[k] = True
                        bottom_cards[n_selected] = hand7[k]
                        n_selected += 1
                        if n_selected == n_bottom:
                            break

            # Phase 3: rebuild deck in a single pass.
            #   [kept hand (7 - n_bottom cards)] | [original tail (deck[7:])] | [bottom cards]
            write_pos = 0
            for k in range(7):
                if not will_bottom[k]:
                    deck[write_pos] = hand7[k]
                    write_pos += 1

            tail_size = deck.size - 7
            for k in range(tail_size):
                deck[write_pos + k] = deck[7 + k]

            for i in range(n_bottom):
                deck[deck.size - n_bottom + i] = bottom_cards[i]

        # --- Evaluate the opening hand ---
        n_land = 0
        n_ramp = 0
        n_bomb = 0
        n_draw = 0

        for k in range(hand_size):
            c = deck[k]
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

    # Fallback: return the last attempted hand regardless of whether it meets requirements
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

        #Draw all cards that are seen until T4 included
        hand = np.empty(7 + len(gameplan) + 1, dtype=np.uint8)  # 7 opening + 1/turn + 1 for T1 draw
        hand[:hand_size] = deck[:hand_size]
        deck_pos = hand_size

        #Default values
        success = True
        fail_turn = -1
        fail_op = -1
        
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
                    
                    for j in range(hand_size): #loop through hand... 
                        if hand[j] == 4: # ...to search for draw spell
                            # Cast draw spell (remove it from hand)
                            hand[j:hand_size-1] = hand[j+1:hand_size]
                            hand_size -= 1
                            
                            # Resolve effect: Draw next card from deck
                            if deck_pos < len(deck):
                                hand[hand_size] = deck[deck_pos]
                                hand_size += 1
                                deck_pos += 1
                            break
                    # Optional: skip regardless of whether Draw was found —
                    # absence of a Draw spell at T1 is not a failure.
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