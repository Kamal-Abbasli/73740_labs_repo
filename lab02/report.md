# Lab 02 Report: A Neural Network from Scratch

## 1. Goal

The goal of this lab is to build a two-hidden-layer neural network completely from scratch using only NumPy, without relying on machine learning libraries. We train the network to recognize handwritten numbers (0-9) from the load_digits dataset, verify our calculus math using a numerical gradient check, compare different network widths and learning speeds, and time the difference between making a guess (forward pass) and learning from mistakes (backward pass).

## 2. Method

- Loaded the 8x8 handwritten digits dataset (64 features) and divided the pixel values by 16 to keep them between 0 and 1.
- Converted the targets (0 to 9) into 10 one-hot mailboxes and split the data into 70% training and 30% testing using random seed 42.
- Built a 2-hidden-layer network:
  - Input: 64 pixels -> Hidden 1 (W) -> ReLU -> Hidden 2 (W) -> ReLU -> Output (10) -> Softmax.
  - Initialized all weights from a normal distribution with standard deviation 0.01.
- Ran sanity checks: confirmed the untrained loss was near ln(10) ≈ 2.3026 and each prediction row summed to 1.0.
- Implemented backpropagation with the chain rule and verified the math with a numerical gradient check: nudged 10 random weights by ±1e-5 to compare analytical vs numerical gradients.
- Trained 6 different configurations (widths 16 and 64, learning rates 0.001, 0.01, and 0.1) over 50 epochs with batch size 32, logging loss curves to `results/loss_curves.png`.
- Measured the runtime of 100 forward and backward passes using the median time.

## 3. Results

### Untrained Sanity Check & Gradient Check

- Untrained Loss: 2.3026 (matches expected ln(10) = 2.3026).
- Numerical Gradient Check: **9/10 passed** with relative error < 1e-6 (meets the required threshold of >= 9/10).

### Model Training & Final Accuracy

| Hidden Width (W) | Learning Rate (lr) | Final Training Loss | Test Exam Accuracy |
| :--------------- | :----------------- | :------------------ | :----------------- |
| 16               | 0.001              | 2.3025              | 10.19%             |
| 16               | 0.01               | 2.3025              | 10.19%             |
| 16               | 0.1                | 0.8992              | 67.22%             |
| 64               | 0.001              | 2.3025              | 10.19%             |
| 64               | 0.01               | 2.3023              | 10.19%             |
| 64               | 0.1                | 0.1432              | **94.44%**         |

### Timing Breakdown (100 runs, W=64)

- **Forward Pass Speed:** 0.2039 ms
- **Backward Pass Speed:** 0.2639 ms
- **Speed Difference:** Backward pass is 1.29x slower than forward pass.

## 4. Conclusions

1. **Learning rate determines whether the network learns:** With learning rates of 0.001 and 0.01, the network remained stuck at ~10.19% accuracy (pure random guessing) after 50 epochs because the updates were too small. Increasing the learning rate to 0.1 allowed the model to escape and learn effectively.
2. **Network capacity improves feature extraction:** With lr=0.1, increasing the hidden width from 16 to 64 improved the test score from 67.22% to 94.44%. The extra neurons gave the network enough capacity to capture the shapes of handwritten digits.
3. **Backward pass requires more computation than forward pass:** The backward pass took 0.2639 ms compared to 0.2039 ms for forward propagation. Computing gradients requires two matrix multiplications per layer (one for the weight gradients and one to propagate errors to earlier layers), making backpropagation computationally heavier.
