# EX.NO:3 Implementation-of-Linear-Regression-Using-Gradient-Descent
# DATE:
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

### Steps involved: 

1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:

Program to implement the linear regression using gradient descent.

Developed by: harithashree

RegisterNumber:  212222230046


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
    #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
    #calculate erros
        error=(predictions-y).reshape(-1,1)
    #update thera using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(error) 
    
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```




## Output:
1. Placememt data:

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/0725e33a-83ac-49b6-a08d-ac7cf652eddf)

2. Salary data:

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/e1af4f86-cb3e-4095-a0f9-31a33e9384ea)

3. Checking the null function:

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/a2206ed1-58eb-407b-a3f0-b1def3d0eeae)

4. Data duplicate:
   
![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/103cce83-e096-4a18-928c-0db04d93e4ad)

5.Print data:


![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/a8900354-1199-4356-a14f-a7c8ff495155)

6. Data status:
   

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/9d00dffb-8507-4be4-859a-1e25975bf0b0)

7. Y-Prediction array:
   

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/ef757539-490f-412f-acc1-4f952be7b4d4)

8. Accuray value:
 

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/44a954fc-f146-45eb-929f-46439cb95a1f)

9. Confusion matrix:
    

![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/e82f2908-cfc3-451a-93da-a1a44c13f72b)

10. Classification Report:


![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/ccdadc3d-7df7-437d-a816-bf7cdd2816fa)

11. Prediction of LR:


![image](https://github.com/haritha-venkat/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121285701/00b464f0-74a9-430d-8dee-5588da581a34)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
