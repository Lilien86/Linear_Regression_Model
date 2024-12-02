# **Linear Regression with PyTorch**

## **Overview**
Implementation of a linear regression model to predict a target variable (`y`) from an input variable (`X`). I show a basic ML process: data preparation, model definition, training, evaluation, and saving.

---

## **Model Details**

- **Architecture**: Single linear layer (`nn.Linear`) 1 input, 1 output feature.
- **Loss Function**: **L1 Loss** (Mean Absolute Error)
- **Optimizer**: **SGD** (Stochastic Gradient Descent) learning rate of 0.01.
- **Training**: 1000 epochs with forward propagation, backpropagation, and weight updates.

---

## **Training and Evaluation**
The training script:
1. Splits data into train (80%) and test (20%) sets.
2. Trains the model using the train set.
3. Evaluates performance on the test set every 100 epochs.

---

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/linear-regression-pytorch.git
   cd linear-regression-pytorch
   ```
2. Diplace model "models/01_pytorch_workflow_model_1.pth" to your project
3. Load model in your code
    ```py
        from model import LinearRegressionModelV2
        import torch

        model = LinearRegressionModelV2()
        model.load_state_dict(torch.load("models/01_pytorch_workflow_model_1.pth"))
        model.eval()
    ```

