# Adaptive Perceptron for Data Streams (Functional Approach)
## Project Overview & Results

---

# Slide 1: Problem Statement & Goal

**Goal:** Build an adaptive perceptron that adapts to a data stream with "Concept Drift" (shifting data).

**The Challenge:**
- Standard models assume static data.
- Our data changes over time (3 distinct Batches with different centers).
- **Architecture:** We avoid complex Classes, using a pure **Functional Approach** for transparency.

**Success Criteria:**
- Detect drift via accuracy drops.
- Recover to > 80% accuracy.

---

# Slide 2: Key Concept - Concept Drift

**Definition:** When the relationship between input data and target labels changes.
**Visual:**
![Concept Drift](concept_drift_infographic.png)

- **Scenario:** Batch 0 is in one location. Batch 1 moves (+0.8, -0.6).
- **Impact:** Old weights no longer work. Accuracy crashes.

---

# Slide 3: The Solution - Adaptive Functional Logic

We implemented two key functions to manage state:

1.  **Decay Schedule:**
    - *Function:* `adapt()` checks `epoch % 5`.
    - *Action:* Returns `new_lr = current_lr * 0.9`.
    - *Visual:* ![Adaptive Learning](adaptive_learning_infographic.png)

2.  **Emergency Reset:**
    - *Condition:* Validation Accuracy < 70%.
    - *Action:* `adapt()` returns `RESET` flag. Main loop calls `initialize_weights()`.
    - *Result:* Model wipes memory to learn the new concept fresh.

---

# Slide 4: Results & Observations

**Performance Timeline:**
- **Batch 0:** Stable accuracy (>90%). LR Decays.
- **Batch 1 (Drift):** Accuracy drops to ~60%. **RESET TRIGGERED**.
- **Recovery:** Model quickly learns new position with high LR.
- **Batch 2:** Minor drift, handled by standard updates or minor reset.

**Conclusion:**
The **Functional Approach** cleanly separates state (weights) from logic (training), making the "Reset" mechanism explicit and easy to trace.
