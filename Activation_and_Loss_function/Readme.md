#Activation Functions:

An activation function is a crucial component in deep learning models.
It performs a nonlinear transformation on the input to get better results on a complex neural network.
The purpose of an activation function is to introduce complexity and non-linearity to the model, enabling it to learn and represent non-linear and complex relationships within the data. Without activation functions, the entire neural network would behave like a linear regression model, regardless of its depth.

Activation functions play a critical role in determining the output of each neuron, thereby shaping the overall behavior and performance of the neural network. Choosing the right activation function for a particular task is an essential part of designing an effective neural network architecture.


## Types of Activation Function:

1.  Sigmoid Activation Function:

   
![image](https://github.com/Recode-Hive/machine-learning-repos/assets/72333364/95e3257c-7c7e-454d-b3ab-f93cbe2aaef2)



    - Formula:  `1/(1+e^-x)`
    - Graph: Smooth, S-shaped curve.
    - Range: (0,1)
    - Suitable for binary classification tasks where the output needs to be interpreted as probabilities.
    - As the range is minimum, prediction would be more accurate.
    - It causes a problem mainly called as vanishing gradient problem which occurs during backpropagation, the gradients can       become extremely small, especially for large positive or negative inputs, which can slow down learning or even cause it to stall.

3. Hyperbolic Tangent (tanh) Activation Function:

![image](https://github.com/Recode-Hive/machine-learning-repos/assets/72333364/3ef16d38-8a51-4f1c-9441-14adc67f231e)


    - Formula: `tan(hx) = (e^x - e^-x)/(e^x + e^-x)`
    - Graph: Also exhibits a smooth, S-shaped curve.
    - Range: (-1, 1)
    - It is also used to predict or to differentiate between two classes but it maps the negative input into negative quantity only 
    - Can suffer from vanishing gradients similar to the sigmoid function.

4. Rectified Linear Unit (ReLU) Activation Function:

   ![image](https://github.com/Recode-Hive/machine-learning-repos/assets/72333364/f5d4ee21-87b7-4e4b-9d99-ce2f252bfd35)


    - Formula: `f(x) = max(0,x) `
    - Range: [0, ∞)
    - Outputs zero for negative inputs and the input value for positive inputs.
    - Overcomes the vanishing gradient problem associated sigmoid and tanh function.
    - Problem associated with it is unbounded on the positive side, which can lead to exploding gradients, especially in deep    
    networks. It also suffers from a problem known as Dying ReLU which is ReLU neurons can sometimes become "dead" during training, meaning they always output zero due to having a negative weight update. This problem particularly occurs when the learning rate is too high, causing a large portion of the neurons to be inactive.

6.  Leaky ReLU Activation Function:

   ![image](https://github.com/Recode-Hive/machine-learning-repos/assets/72333364/fbf2dab1-462d-45da-9534-be95a1de08c1)


    - Formula: `f(x) = {x , if(x>0) and αx , if(x<=0)}`
    - Range: (-∞, ∞)
    - Does not output zero for negative inputs as in ReLU but do make all negative inputs value near to zero which solves the major issue of ReLU activation function.
    - It also solves the problem of Dying ReLU as discussed in ReLU Activation Function.
    - Introduces an additional hyperparameter (α) that needs to be tuned, although often a small value like 0.01 suffices.

8. Softmax Activation Function:

   ![image](https://github.com/Recode-Hive/machine-learning-repos/assets/72333364/07da0843-df3e-400f-8d3c-4cd44f204c74)


    - the softmax basically gives value to the input variable according to their weight.
    - Range: (0, 1) for each output, with the sum of outputs equal to 1.
    - Primarily used in the output layer of a neural network for multi-class classification problems.
    - Softmax output is dependent on the values of all other outputs, making it sensitive to changes in other predictions. This can make training more complex, especially in multi-label classification scenarios.
    


# Loss Functions:

The loss function is used to measure how good or bad the model is performing.It quantifies how well the model's predictions match the actual target values (labels) in the training data.The goal during training is to minimize this loss function, thereby improving the model's performance.


## Types of Loss Function:

Loss functions are mainly classified into two different categories that are Classification loss and Regression Loss.
Classification loss is the case where the aim is to predict the output from the different categorical values where as if the problem is regression like predicting the continuous values then Regression loss is used.

1. Regression Loss Functions:
    - Mean Squared Error(MSE): Calculates the average squared difference between predicted and actual values. 
    - Mean Absolute Error(MAE): Computes the average absolute difference between predicted and actual values.

2. Classification Loss Functions:
    - Binary Cross-Entropy Loss (Log Loss): Used in binary classification tasks. It measures the dissimilarity between the true binary labels and the predicted probabilities.
    - Categorical Cross-Entropy Loss: Generalizes binary cross-entropy to multi-class classification tasks. It is used when there are multiple classes to predict.
    - Sparse Categorical Cross-Entropy Loss: Similar to categorical cross-entropy but is more efficient when dealing with sparse target labels, commonly used in language modeling tasks.
    - Hinge Loss: Used in maximum-margin classification, particularly in support vector machines (SVMs).
    - Sigmoid Cross-Entropy Loss: Similar to binary cross-entropy but used when the output of the model needs to be transformed by a sigmoid activation function.


