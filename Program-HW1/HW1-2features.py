import numpy as np

# import data file
file_path = "TWOCLASS.dat"

try:
    with open(file_path, "r") as file:
        content = file.read()
        data = content.split()
        # print(data)
except FileNotFoundError:
    print("File not found!")

column = ['f1', 'f2', 'f3', 'f4', 'class label']

# drop the first 6 elements from list 'data'
data = data[6:]

data_processed = []

for i in range(0, len(data), 5):
    data_processed.append([float(data[i]), float(data[i+1]), float(data[i+2]), \
                           float(data[i+3]), int(data[i+4])])

# print length of data
print("Length of data: ", len(data_processed))

print('Using features 1 and 2')
# Step 1: Estimate Parameters (Mean vectors and Covariance matrices)
def estimate_parameters(data):
    num_features = 2  # Considering only features 1 and 2
    num_classes = max(data, key=lambda x: x[-1])[-1]  # Assuming class labels are 1-indexed

    mean_vectors = {i: np.zeros(num_features) for i in range(1, num_classes + 1)}
    covariance_matrices = {i: np.zeros((num_features, num_features)) for i in range(1, num_classes + 1)}
    class_counts = {i: 0 for i in range(1, num_classes + 1)}

    # Calculate the sum of feature values for each class
    for row in data:
        class_label = int(row[-1])
        class_counts[class_label] += 1
        for i in range(num_features):
            mean_vectors[class_label][i] += row[i]

    # Calculate the mean vectors
    for class_label in mean_vectors:
        mean_vectors[class_label] /= class_counts[class_label]

    # Calculate the covariance matrices
    for row in data:
        class_label = int(row[-1])
        x_minus_mean = row[:num_features] - mean_vectors[class_label]
        x_minus_mean = x_minus_mean.reshape((-1, 1))  # Convert to column vector
        covariance_matrices[class_label] += np.dot(x_minus_mean, x_minus_mean.T)

    for class_label in covariance_matrices:
        covariance_matrices[class_label] /= (class_counts[class_label] - 1)

    return mean_vectors, covariance_matrices

# Step 2: Minimum Risk Bayes Decision Theoretic Classifier
def minimum_risk_classifier(test_sample, mean_vectors, covariance_matrices, prior_probabilities):
    num_classes = len(mean_vectors)
    posterior_probabilities = [0] * num_classes

    for class_label in range(1, num_classes + 1):
        mean_vector = np.array(mean_vectors[class_label])
        covariance_matrix = np.array(covariance_matrices[class_label])

        # Convert the test_sample to a NumPy array and consider only features 1 and 2
        test_sample_arr = np.array(test_sample[:2])

        # Calculate (x - mean)
        x_minus_mean = test_sample_arr - mean_vector

        # Calculate the determinant of the covariance matrix
        det_covariance = np.linalg.det(covariance_matrix)

        # Calculate the inverse of the covariance matrix
        inv_covariance = np.linalg.inv(covariance_matrix)

        # Calculate the Mahalanobis distance squared
        mahalanobis_dist_sq = np.dot(x_minus_mean, np.dot(inv_covariance, x_minus_mean))

        # Calculate the log-likelihood and posterior probability without using math.log
        log_likelihood = -0.5 * (mahalanobis_dist_sq + np.log(det_covariance))
        posterior_probabilities[class_label - 1] = np.log(prior_probabilities[class_label]) + log_likelihood

    # Choose the class with the highest posterior probability as the predicted label
    predicted_label = np.argmax(posterior_probabilities) + 1

    return predicted_label

# Step 3: 10% Cross Validation
def cross_validation(data):
    num_samples_per_class = len(data) // 2
    fold_size = num_samples_per_class // 10  # Each fold will have 10 samples per class
    total_samples = len(data)
    total_misclassifications = 0
    predicted_labels = []  # List to store the predicted labels for all test samples

    for i in range(10):
        start_idx_class1 = i * fold_size
        end_idx_class1 = (i + 1) * fold_size
        start_idx_class2 = num_samples_per_class + start_idx_class1
        end_idx_class2 = num_samples_per_class + end_idx_class1

        test_set = data[start_idx_class1:end_idx_class1] + data[start_idx_class2:end_idx_class2]
        train_set = data[:start_idx_class1] + data[end_idx_class1:start_idx_class2] + data[end_idx_class2:]

        mean_vectors, covariance_matrices = estimate_parameters(train_set)

        misclassifications = 0
        fold_predicted_labels = []  # List to store the predicted labels for the current fold

        print(f"\nFold {i+1} Results:")
        for sample in test_set:
            test_sample = sample[:-1]  # Exclude the class label from the test sample
            true_label = int(sample[-1])
            predicted_label = minimum_risk_classifier(test_sample, mean_vectors, covariance_matrices, prior_probabilities)
            fold_predicted_labels.append(predicted_label)  # Store the predicted label for the current sample

            if predicted_label != true_label:
                misclassifications += 1

            # Display classification information for the current test sample
            print(f"Fold {i+1} - True Label: {true_label}, Predicted Label: {predicted_label}")

        predicted_labels.extend(fold_predicted_labels)  # Add the predicted labels for the current fold to the list
        fold_accuracy = 1 - (misclassifications / (2 * fold_size))
        total_misclassifications += misclassifications

        # Display results for the current fold
        print(f"Accuracy: {fold_accuracy:.2f}")
        print(f"Misclassifications: {misclassifications}/{2 * fold_size}")

        confusion_matrix = [[0, 0], [0, 0]]
        for true_label, predicted_label in zip([1] * fold_size + [2] * fold_size, fold_predicted_labels):
            confusion_matrix[true_label - 1][predicted_label - 1] += 1

        display_results(mean_vectors, covariance_matrices, confusion_matrix)

    overall_accuracy = 1 - (total_misclassifications / total_samples)
    print("\nOverall Results:")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")
    print(f"Overall Misclassifications: {total_misclassifications}/{total_samples}")

    avg_accuracy = 1 - (total_misclassifications / total_samples)
    return avg_accuracy, predicted_labels

# Step 4: Display Results (Mean vectors, Covariance matrices, Confusion matrix, Error estimates, Classification information)
def display_results(mean_vectors, covariance_matrices, confusion_matrix):
    print("Mean Vectors:")
    for class_label, mean_vector in mean_vectors.items():
        print(f"Class {class_label}: {mean_vector}")

    print("\nCovariance Matrices:")
    for class_label, covariance_matrix in covariance_matrices.items():
        print(f"Class {class_label}:")
        for row in covariance_matrix:
            print(row)

    print("\nConfusion Matrix:")
    for row in confusion_matrix:
        print(row)

    total_samples = sum(sum(row) for row in confusion_matrix)
    correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    error_rate = (total_samples - correct_predictions) / total_samples
    print(f"\nError Rate: {error_rate:.2f}")
    print('\n----------------------------------------')

# Step 5: Main code
# Example data
data = data_processed

# Assuming equal prior probabilities for each class (not hard-coded)
num_classes = max(data, key=lambda x: x[-1])[-1]
prior_probabilities = {i: 1 / num_classes for i in range(1, num_classes + 1)}

# Perform cross-validation and display the results
avg_accuracy, predicted_labels = cross_validation(data)
print(f"\nAverage Accuracy: {avg_accuracy:.2f}")

# Estimate mean vectors and covariance matrices using the entire data
mean_vectors, covariance_matrices = estimate_parameters(data)

# Calculate confusion matrix using the entire data
confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
for sample in data:
    true_label = int(sample[-1])
    test_sample = sample[:-1]
    predicted_label = minimum_risk_classifier(test_sample, mean_vectors, covariance_matrices, prior_probabilities)
    confusion_matrix[true_label - 1][predicted_label - 1] += 1

print("\nOverall Results:")
print("Mean Vectors:")
for class_label, mean_vector in mean_vectors.items():
    print(f"Class {class_label}: {mean_vector}")

print("\nCovariance Matrices:")
for class_label, covariance_matrix in covariance_matrices.items():
    print(f"Class {class_label}:")
    for row in covariance_matrix:
        print(row)

print("\nConfusion Matrix:")
for row in confusion_matrix:
    print(row)

total_samples = sum(sum(row) for row in confusion_matrix)
correct_predictions = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
error_rate = (total_samples - correct_predictions) / total_samples
print(f"\nError Rate: {error_rate:.2f}")