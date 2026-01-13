# AN-simulator

Monte Carlo simulation framework for optimizing Commander (EDH) deck composition
with Adrix and Nev, Twincasters as commander. Features a predefined gameplan, with
detailed mulligan logic and failure diagnostics.

This project focuses on evaluating how card-type distributions (Lands, Ramp, Bombs, Draw)
affect the probability of successfully executing a multi-turn gameplan.

Check the primer of the deck for more information: https://moxfield.com/decks/YOujHfMqzkqnsIXzll97FQ

---

## 📌 Features

- Numba-accelerated Monte Carlo simulation
- Custom mulligan logic (free mulligan + priority bottoming)
- Turn-by-turn gameplan execution
- Detailed failure diagnostics (turn + operation)
- Grid search optimization over deck compositions
- Sensitivity analysis of card counts
---

## 🎴 Deck Model

- Deck size: 99 cards
- Card encoding:
  - `0` → Other
  - `1` → Land
  - `2` → Ramp
  - `3` → Bomb
  - `4` → Draw

- Gameplan is defined as per-turn required card plays
- Optional Draw cards can replace themselves when played

---

## 🚀 Getting Started

### 1. Clone the repository
```
git clone https://github.com/enarfr/AN-simulator.git
cd AN-simulator
```

### 2. Create a Conda environment (recommended)
```
conda env create -f environment.yml
```

### 3. Run the notebook

```
jupyter lab
```
Open notebook: main-simulator.ipynb and execute cells sequentially

## 📊 Output
The simulator reports:
* Overall success rate
* Failure rate
* Failure breakdown by turn and operation
* Mulligan usage statistics

## 🧠 Performance Notes
* Core simulation is compiled with Numba
* First execution includes compilation overhead
* Subsequent runs are significantly faster
* Parallel execution uses prange

## ⚠️ Disclaimer
This tool evaluates only the modeled gameplan.
It does not account for interaction, mana color constraints,
opponent disruption, or alternative lines of play.

Use results as decision support, not absolute truth.

