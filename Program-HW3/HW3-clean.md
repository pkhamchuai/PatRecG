# Computer Assignment 3


```python
import pandas as pd
import numpy as np

# open the txt file and read its contents
with open('ionosphere.txt', 'r', encoding='ISO-8859-1') as file:
    data = file.read()

# remove first 2 letters
data = data[2:]

# remove '\x00' from the data
data = data.replace('\x00', '')

# create a dataframe from the data
df = pd.DataFrame([row.split() for row in data.split('\n')])

# drop the row with odd index
df = df.drop(df.index[1::2])
# drop the last row
df = df.drop(df.index[-1])

# firt row is the column names
df.columns = df.iloc[0]
# drop the first row
df = df.drop(df.index[0])

column_name = [f"f{i}" for i in range(1, 35)] + ['class']

# rename the columns
df.columns = column_name

# convert the class column to binary
df['class'] = df['class'].replace(['g', 'b'], [1, 2]) # g = 1, b = 2

# convert the data type to float
df = df.astype(float)

# reset the index
df = df.reset_index(drop=True)

df.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1</th>
      <th>f2</th>
      <th>f3</th>
      <th>f4</th>
      <th>f5</th>
      <th>f6</th>
      <th>f7</th>
      <th>f8</th>
      <th>f9</th>
      <th>f10</th>
      <th>...</th>
      <th>f26</th>
      <th>f27</th>
      <th>f28</th>
      <th>f29</th>
      <th>f30</th>
      <th>f31</th>
      <th>f32</th>
      <th>f33</th>
      <th>f34</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.99539</td>
      <td>-0.05889</td>
      <td>0.85243</td>
      <td>0.02306</td>
      <td>0.83398</td>
      <td>-0.37708</td>
      <td>1.00000</td>
      <td>0.03760</td>
      <td>...</td>
      <td>-0.51171</td>
      <td>0.41078</td>
      <td>-0.46168</td>
      <td>0.21266</td>
      <td>-0.34090</td>
      <td>0.42267</td>
      <td>-0.54487</td>
      <td>0.18641</td>
      <td>-0.45300</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.00000</td>
      <td>-0.18829</td>
      <td>0.93035</td>
      <td>-0.36156</td>
      <td>-0.10868</td>
      <td>-0.93597</td>
      <td>1.00000</td>
      <td>-0.04549</td>
      <td>...</td>
      <td>-0.26569</td>
      <td>-0.20468</td>
      <td>-0.18401</td>
      <td>-0.19040</td>
      <td>-0.11593</td>
      <td>-0.16626</td>
      <td>-0.06288</td>
      <td>-0.13738</td>
      <td>-0.02447</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.00000</td>
      <td>-0.03365</td>
      <td>1.00000</td>
      <td>0.00485</td>
      <td>1.00000</td>
      <td>-0.12062</td>
      <td>0.88965</td>
      <td>0.01198</td>
      <td>...</td>
      <td>-0.40220</td>
      <td>0.58984</td>
      <td>-0.22145</td>
      <td>0.43100</td>
      <td>-0.17365</td>
      <td>0.60436</td>
      <td>-0.24180</td>
      <td>0.56045</td>
      <td>-0.38238</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.00000</td>
      <td>-0.45161</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>0.71216</td>
      <td>-1.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>0.90695</td>
      <td>0.51613</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>-0.20099</td>
      <td>0.25682</td>
      <td>1.00000</td>
      <td>-0.32382</td>
      <td>1.00000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.00000</td>
      <td>-0.02401</td>
      <td>0.94140</td>
      <td>0.06531</td>
      <td>0.92106</td>
      <td>-0.23255</td>
      <td>0.77152</td>
      <td>-0.16399</td>
      <td>...</td>
      <td>-0.65158</td>
      <td>0.13290</td>
      <td>-0.53206</td>
      <td>0.02431</td>
      <td>-0.62197</td>
      <td>-0.05707</td>
      <td>-0.59573</td>
      <td>-0.04608</td>
      <td>-0.65697</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



## Define variables, constants


```python
N = len(df)
print(f"Number of samples: {N}")
```

    Number of samples: 351


# Feature selection

## T-Test at 99% confident


```python
# import stats
import scipy.stats as stats

def ttest(df, feature):
    """
    Perform t-test on the given feature and return the t-statistic and p-value
    """

    # group the dataframe by class
    grouped = df.groupby("class")
    # get the groups
    group1 = grouped.get_group(1)[feature]
    group2 = grouped.get_group(2)[feature]
    print(f"Group 1 Mean: {group1.mean()}")
    print(f"Group 1 Variance: {group1.var()}")
    print(f"Group 2 Mean: {group2.mean()}")
    print(f"Group 2 Variance: {group2.var()}")

    # if mean and variance are equal, then the t-statistic is 0
    # and the p-value is 1
    # so, we can return False
    if group1.mean() == group2.mean() and group1.var() == group2.var():
        print("The null hypothesis is accepted")
        print("The feature is rejected")
        return False        

    # perform t-test
    # variance is unknown
    N1 = len(group1)
    N2 = len(group2)
    print(f"N1: {N1}")
    print(f"N2: {N2}")
    if N1 == N2:
        dof = 2*N1 - 2
        print("N1 == N2")
        print("Performing t-test...")
        s_squared = (group1.var() + group2.var())/(2*N1 - 2)
        s = s_squared**0.5
        q = (group1.mean() - group2.mean()) / (s * (2/N1))**0.5
        
        # p-value
        p_value = 1 - stats.t.cdf(abs(q), df=dof)
        print(f"p-value: {p_value}")

        # confidence interval
        ci = stats.t.ppf(0.99, dof)
        print(f"Confidence Interval (ci): {ci}")
        print(f"q: {q}")
        print(f"q in (-ci, ci): {-ci <= q <= ci}")
        if -ci <= q <= ci:
            print("The null hypothesis is accepted")
            print("The feature is rejected")
            return False
        else:
            print("The null hypothesis is rejected")
            print("The feature is accepted")
            return True

    else:
        print("N1 != N2")
        print("Performing t-test...")
        dof = N1 + N2 - 2
        s_squared = (group1.var() + group2.var())/(N1 + N2 - 2)
        s = s_squared**0.5
        q = (group1.mean() - group2.mean()) / (s * ((1/N1) + (1/N2))**0.5)
        
        # p-value
        p_value = 1 - stats.t.cdf(abs(q), df=dof)
        print(f"p-value: {p_value}")

        # confidence interval
        ci = stats.t.ppf(0.99, dof)
        print(f"Confidence Interval (ci): {ci}")
        print(f"q: {q}")
        print(f"q in (-ci, ci): {-ci <= q <= ci}")
        if -ci <= q <= ci:
            print("The null hypothesis is accepted")
            print("The feature is rejected")
            return False
        else:
            print("The null hypothesis is rejected")
            print("The feature is accepted")
            return True

```


```python
# calculate t-statistic and p-value for each feature
feature_accepted = []

print(f"Degrees of Freedom: {N - 2}")
print("T-Test Results\n")

for feature in df.columns[:-1]:
    print(f"Feature: {feature}")
    feature_accepted.append(ttest(df, feature))
    
    print()

```


```python
# the accepted features are 
accepted_features = [feature for feature, accepted in zip(df.columns[:-1], feature_accepted) if accepted]
print(f"Accepted Features: {accepted_features}")

print(f"Rejected Features: {[feature for feature in df.columns[:-1] if feature not in accepted_features]}")
```

    Accepted Features: ['f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f25', 'f27', 'f28', 'f29', 'f31', 'f32', 'f33', 'f34']
    Rejected Features: ['f2', 'f24', 'f26', 'f30']


# Linear classifier


```python
class LinearClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.T = num_iterations
        self.w = None
        self.b = None
        self.labels = None
        
    def fit(self, X, y):        
        # take the labels from y
        self.labels = np.unique(y)
        decision_boundary = (self.labels[0] + self.labels[1]) / 2

        # initialize the weights and bias to zeros
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        # gradient descent
        for i in range(self.T):
            # calculate the predicted values
            y_pred = np.dot(X, self.w) + self.b
            
            # calculate the gradients/cost function
            dw = (1/X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1/X.shape[0]) * np.sum(y_pred - y)
            
            # update the weights and bias if misclassified
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # check termination condition, if satisfied, break
            if np.linalg.norm(dw) < 1e-4:
                # print(f"Terminated at iteration {i}")
                break

            
    def predict(self, X):
        # calculate the predicted values
        y_pred = np.dot(X, self.w) + self.b
        
        # convert the predicted values to binary
        decision_boundary = (self.labels[0] + self.labels[1]) / 2
        y_pred_binary = np.where(y_pred < decision_boundary, self.labels[0], self.labels[1])
        
        return y_pred_binary

```


```python
def cross_validate(df, features, target):
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    # calculate the number of samples in 10% of the dataframe
    n_samples = int(len(df) * 0.1)
    
    # initialize the accuracy list
    accuracy_list = []
    
    # loop through each split
    for i in range(10):
        # calculate the start and end indices for the test set
        start_index = i * n_samples
        end_index = (i + 1) * n_samples
        
        # split the data into train and test sets
        X_test = df.iloc[start_index:end_index][features]
        X_train = pd.concat([df.iloc[:start_index][features], df.iloc[end_index:][features]])
        y_train = pd.concat([df.iloc[:start_index][target], df.iloc[end_index:][target]])
        y_test = df.iloc[start_index:end_index][target]

        # initialize the linear classifier
        clf = LinearClassifier()
        
        # fit the classifier on the train set
        clf.fit(X_train, y_train)
        
        # predict the target values for the test set
        y_pred = clf.predict(X_test)
        
        # calculate the accuracy of the classifier
        accuracy = sum(y_pred == y_test) / len(y_test)
        
        # append the accuracy to the accuracy list
        accuracy_list.append(accuracy)

        # add other classifiers here
    
    # calculate the mean accuracy
    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    
    return mean_accuracy

```

# The results


```python
target_column = 'class'
acc_list = []

# using all features
print("Using all features")

# collect time execution
import time
start = time.time()
accuracy = cross_validate(df, df.columns[:-1], target_column)
end = time.time()
print(f"Mean accuracy: {accuracy}\n")
acc_list.append(['all', accuracy, end-start])

# using accepted features
print("Using accepted features")
start = time.time()
accuracy = cross_validate(df, accepted_features, target_column)
end = time.time()
print(f"Mean accuracy: {accuracy}")
acc_list.append(['accepted', accuracy, end-start])
```

    Using all features
    Mean accuracy: 0.8142857142857143
    
    Using accepted features
    Mean accuracy: 0.8342857142857142



```python
# using rejected features
print("Using rejected features")
rejected_features = [feature for feature in df.columns[:-1] if feature not in accepted_features]
start = time.time()
accuracy = cross_validate(df, rejected_features, target_column)
end = time.time()
print(f"Mean accuracy: {accuracy}")
acc_list.append(['rejected', accuracy, end-start])
```

    Using rejected features
    Mean accuracy: 0.642857142857143



```python
# using one feature at a time
# loop through the features and calculate the accuracy
for feature in df.columns[:-1]:
    start = time.time()
    accuracy = cross_validate(df, [feature], target_column)
    end = time.time()
    acc_list.append([feature, accuracy, end-start])
```


```python
# create a dataframe from the accuracy list
acc_df = pd.DataFrame(acc_list, columns=['feature', 'accuracy', 'time'])
# convert accuracy to percent and round to 2 decimal places
acc_df['accuracy'] = round(acc_df['accuracy'] * 100, 2)
# convert time to 2 decimal places
acc_df['time'] = round(acc_df['time'], 2)
# show acc_df sorted by accuracy
acc_df.sort_values(by='accuracy', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>accepted</td>
      <td>83.43</td>
      <td>29.28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>f5</td>
      <td>82.00</td>
      <td>5.37</td>
    </tr>
    <tr>
      <th>0</th>
      <td>all</td>
      <td>81.43</td>
      <td>29.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f3</td>
      <td>76.86</td>
      <td>4.84</td>
    </tr>
    <tr>
      <th>16</th>
      <td>f14</td>
      <td>69.43</td>
      <td>4.38</td>
    </tr>
    <tr>
      <th>9</th>
      <td>f7</td>
      <td>68.00</td>
      <td>3.56</td>
    </tr>
    <tr>
      <th>10</th>
      <td>f8</td>
      <td>67.71</td>
      <td>3.46</td>
    </tr>
    <tr>
      <th>33</th>
      <td>f31</td>
      <td>67.14</td>
      <td>5.51</td>
    </tr>
    <tr>
      <th>31</th>
      <td>f29</td>
      <td>66.86</td>
      <td>5.35</td>
    </tr>
    <tr>
      <th>25</th>
      <td>f23</td>
      <td>66.00</td>
      <td>3.89</td>
    </tr>
    <tr>
      <th>11</th>
      <td>f9</td>
      <td>66.00</td>
      <td>4.92</td>
    </tr>
    <tr>
      <th>27</th>
      <td>f25</td>
      <td>65.14</td>
      <td>3.97</td>
    </tr>
    <tr>
      <th>18</th>
      <td>f16</td>
      <td>65.14</td>
      <td>2.92</td>
    </tr>
    <tr>
      <th>14</th>
      <td>f12</td>
      <td>64.86</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>f4</td>
      <td>64.29</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f2</td>
      <td>64.29</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rejected</td>
      <td>64.29</td>
      <td>5.36</td>
    </tr>
    <tr>
      <th>32</th>
      <td>f30</td>
      <td>64.29</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f1</td>
      <td>64.29</td>
      <td>4.85</td>
    </tr>
    <tr>
      <th>21</th>
      <td>f19</td>
      <td>64.29</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>24</th>
      <td>f22</td>
      <td>64.29</td>
      <td>5.17</td>
    </tr>
    <tr>
      <th>35</th>
      <td>f33</td>
      <td>64.00</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>34</th>
      <td>f32</td>
      <td>64.00</td>
      <td>3.41</td>
    </tr>
    <tr>
      <th>30</th>
      <td>f28</td>
      <td>64.00</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>29</th>
      <td>f27</td>
      <td>64.00</td>
      <td>4.93</td>
    </tr>
    <tr>
      <th>28</th>
      <td>f26</td>
      <td>64.00</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>13</th>
      <td>f11</td>
      <td>64.00</td>
      <td>4.09</td>
    </tr>
    <tr>
      <th>26</th>
      <td>f24</td>
      <td>64.00</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>22</th>
      <td>f20</td>
      <td>64.00</td>
      <td>3.24</td>
    </tr>
    <tr>
      <th>20</th>
      <td>f18</td>
      <td>64.00</td>
      <td>5.43</td>
    </tr>
    <tr>
      <th>19</th>
      <td>f17</td>
      <td>64.00</td>
      <td>5.39</td>
    </tr>
    <tr>
      <th>8</th>
      <td>f6</td>
      <td>64.00</td>
      <td>3.11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>f10</td>
      <td>64.00</td>
      <td>5.01</td>
    </tr>
    <tr>
      <th>36</th>
      <td>f34</td>
      <td>64.00</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>15</th>
      <td>f13</td>
      <td>62.00</td>
      <td>4.43</td>
    </tr>
    <tr>
      <th>23</th>
      <td>f21</td>
      <td>61.71</td>
      <td>3.96</td>
    </tr>
    <tr>
      <th>17</th>
      <td>f15</td>
      <td>61.43</td>
      <td>5.49</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
