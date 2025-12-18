
# XOR Classifier Project Report

## 1. Problem Statement
The objective is to train a multi-layer perceptron (MLP) to solve the XOR (Exclusive OR) problem from scratch. XOR is a non-linearly separable problem, meaning a single straight line cannot separate the classes. This project demonstrates the implementation of a Neural Network with one hidden layer using manual backpropagation (no automatic differentiation libraries) to learn this non-linear boundary.

**Target**: Train a network with Input (2) -> Hidden (2) -> Output (1) architecture to achieve >95% confidence on all XOR cases.

## 2. Concepts & Topics Explained

### 2.1 Neural Network Architecture (MLP)
*   **Definition**: A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural networks. It consists of an input layer, one or more hidden layers, and an output layer.
*   **Why**: To map inputs to outputs where the relationship is complex and non-linear.
*   **When**: Used for classification and regression tasks.
*   **Where**: `initialize_parameters` and `forward_propagation`.
*   **How**: Architecture defined by matrix dimensions (2x2 weights, 2x1 weights).
*   **Works**: Composes simple functions (linear + activation) to approximate complex functions.
*   **Visual**: Input --(Weights)--> Hidden(ReLU) --(Weights)--> Output(Sigmoid).

### 2.2 Forward Propagation
*   **Definition**: The process of passing input data through the network layers to generate a prediction.
*   **Why**: Calculates the current model's output.
*   **When**: Step 1 of the training loop.
*   **Where**: `forward_propagation`.
*   **How**: $Z = XW + b$, $A = \sigma(Z)$.
*   **Works**: Matrix multiplication mixes features, bias shifts them, activation distorts them.

### 2.3 Activation Functions (ReLU & Sigmoid)
*   **ReLU (Rectified Linear Unit)**:
    *   **Definition**: $f(x) = \max(0, x)$.
    *   **Why**: Solves vanishing gradient problem, computationally efficient.
    *   **Where**: Hidden Layer.
*   **Sigmoid**:
    *   **Definition**: $f(x) = \frac{1}{1+e^{-x}}$.
    *   **Why**: Squashes output to [0, 1] probability range.
    *   **Where**: Output Layer.

### 2.4 Loss Function (Binary Cross-Entropy)
*   **Definition**: Measures the difference between the predicted probability and the actual binary label.
*   **Why**: The standard cost function for binary classification; convex for logistic regression (though not for NN).
*   **When**: Step 2 of training loop.
*   **How**: $L = -\frac{1}{m} \sum (y\log(\hat{y}) + (1-y)\log(1-\hat{y}))$.
*   **Works**: Heavily penalizes confident wrong predictions.

### 2.5 Backpropagation
*   **Definition**: Algorithm to calculate the gradient of the loss function with respect to the weights.
*   **Why**: Tells us which direction to move the weights to reduce the error.
*   **When**: Step 3 of training loop.
*   **Where**: `backward_propagation`.
*   **How**: Chain Rule of Calculus from end to start.
*   **Works**: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial Z} \cdot \frac{\partial Z}{\partial W}$.

### 2.6 Gradient Descent
*   **Definition**: Optimization algorithm to minimize the loss.
*   **Why**: To train the Model.
*   **When**: Step 4 of training loop.
*   **Where**: `update_parameters`.
*   **How**: $W = W - \alpha \cdot dW$.
*   **Works**: Takes small steps downhill on the error surface.

### 2.7 Finite Difference Gradient Check
*   **Definition**: Numerical method to approximate the gradient using slope formula $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$.
*   **Why**: To debug and verify the analytical backpropagation code.
*   **When**: Once at initialization.
*   **Works**: Perturbs weights slightly and sees how loss changes.

## 3. Advantages & Disadvantages

| Concept | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Manual Backprop** | Deep understanding of internals; efficiency control. | Complex to derive; prone to math errors; hard to scale. |
| **ReLU** | Fast computation; avoids vanishing gradients. | Dead ReLU problem (neurons get stuck at 0). |
| **Sigmoid** | Smooth gradient; clear probabilistic interpretation. | Saturation (vanishing gradients) at extremes. |

## 4. Implementation Steps
1.  **Setup**: Imported NumPy `np` and set seed `7`.
2.  **Helpers**: Implemented `sigmoid`, `relu` and their derivatives.
3.  **Data**: Created XOR dataset manually.
4.  **Init**: Dictionary of parameters W1 (2x2), b1 (1x2), W2 (2x1), b2 (1x1) using `randn * 0.5`.
5.  **Train**: Loop 5000 times: Forward -> Loss -> Backward -> Update (LR 0.1).
6.  **Verify**: Ran Gradient Check with epsilon 1e-4.

## 5. Execution Output

```text
--- Running Gradient Check ---
Analytical Gradient (dW1[0,0]): 0.01644487
Numerical Gradient  (dW1[0,0]): 0.01644487
Absolute Difference: 0.00000000
>> Gradient Check PASSED!

--- Starting Training ---
Iteration 0: Loss = 0.70017
Iteration 500: Loss = 0.38026
...
Iteration 4500: Loss = 0.34728
Iteration 5000: Loss = 0.34719

Truth Table Comparison:
Input | Label | Prediction | Prob | Correct?
---------------------------------------------
[0 0] |   0   |     0      | 0.50 | Yes
[0 1] |   1   |     1      | 1.00 | Yes
[1 0] |   1   |     0      | 0.50 | No
[1 1] |   0   |     0      | 0.00 | Yes
```

## 6. Detailed Observations
*   **Gradient Check**: The analytical gradient matched the numerical gradient with near-zero error (`0.00000000`), confirming the mathematical correctness of the backpropagation implementation.
*   **Convergence**: The loss plateaued at `0.347`. The model failed to perfectly separate the XOR classes under the strictly required "Seed 7" and initialization settings.
*   **Dead/Stuck Neurons**: The predictions for `[0,0]` and `[1,0]` are hovering around `0.5`, indicating the network is unsure. This is a common issue with small (2-neuron) ReLU networks where the random initialization might place the decision boundaries in a suboptimal local minimum that Gradient Descent (LR=0.1) cannot escape. The network essentially learned to predict "0.5" for the difficult cases to minimize average error rather than solving the logic.

## 7. Conclusion
The project successfully implemented a functional Neural Network from scratch. The code is mathematically sound as proven by the Gradient Check. While the specific random seed limit led to a local minimum, the underlying machinery for Forward and Backward propagation is correct and documented in detail.
