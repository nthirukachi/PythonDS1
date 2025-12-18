# Research Briefing: Manual XOR Classifier

## Slide 1: The Non-Linear Challenge
*   **The Problem**: We needed to verify if a simple Neural Network could solve the XOR problem (Exclusive OR) without using high-level libraries like PyTorch or TensorFlow.
*   **Why it matters**: XOR is the "Hello World" of non-linear deep learning. A straight line cannot separate (0,0) and (1,1) from (0,1) and (1,0). It requires the network to "bend" the space.
*   **The Approach**: We built a 2-layer MLP (Multi-Layer Perceptron) purely in NumPy, implementing the calculus of backpropagation by hand.

## Slide 2: Inside the Engine (Implementation)
*   **Architecture**:
    *   **Input**: 2 Features
    *   **Hidden Layer**: 2 Neurons with **ReLU** activation (The switch that adds non-linearity).
    *   **Output Layer**: 1 Neuron with **Sigmoid** activation (The probability gate).
*   **Visual**:
    ![Neural Network Diagram](xor_nn_architecture_1766074486858.png)

## Slide 3: The Learning Process
![Training Flowchart](xor_training_flow_1766074523203.png)

## Slide 4: Execution Results
*   **Sanity Check**: The gradient check passed with absolute error `0.00000000`. The math is solid.
*   **Training Dynamics**:
    *   **Iterations**: 5,000
    *   **Learning Rate**: 0.1
    *   **Final Loss**: 0.347
*   **Outcome**: The model *struggled* with the specific initialization seed (7).
    *   Separated [0,1] and [1,1] perfectly.
    *   Got stuck on [0,0] vs [1,0], predicting ~0.50 probability for both.

## Slide 4: Key Insights & Takeaways
*   **The "Dead ReLU" Risk**: With only 2 hidden neurons, if initialization pushes them to the negative zone early, they die (gradient becomes 0).
*   **Sensitivity**: Small networks are highly sensitive to random seed initialization. "Seed 7" proved to be a difficult starting point.
*   **Pedagogical Value**: Despite the local minimum, the code serves as a perfect reference for understanding:
    1.  How dimensions change during Forward Prop.
    2.  How error flows backward during Backprop.
    3.  How to debug networks using Gradient Checking.

## Slide 5: Next Steps (Recommendation)
*   **To solve XOR perfectly**:
    1.  Increase Hidden Neurons (e.g., 4 neurons provides robust redundancy).
    2.  Use a different initialization strategy (e.g., He Initialization rather than random * 0.5).
    3.  Use Leaky ReLU to prevent dead neurons.
*   **Conclusion**: Validated the algorithm; identified constraints of the minimal architecture.
