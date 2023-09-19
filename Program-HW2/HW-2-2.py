import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# grab the name of files in the directory 'chrom'
# excluding the file with 'html' extension
# and store the names in a list

chrom_files = glob.glob('chrom/*')
chrom_files = [x for x in chrom_files if 'html' not in x]
print(f'# of files: {len(chrom_files)}')

# files ending with a are training data
# files ending with b are testing data

# list of training data
train_files = [x for x in chrom_files if 'a' in x]
print(f'# of training files: {len(train_files)}')

# list of testing data
test_files = [x for x in chrom_files if 'b' in x]
print(f'# of testing files:  {len(test_files)}')

# function that reads the contents of a file
# and returns a list of lines
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # for each line, take the string after '\t' and before '\n' as data
    # the second number is label
    data = [x.split('\t')[1].split('\n')[0] for x in lines]

    return data

def read_label(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # for each line, take the string after '\t' and before '\n' as data
    # the second number is label
    labels = [x.split('\t')[0] for x in lines]
    # split labels into a list of lists
    labels = [x.split(' ')[2] for x in labels]

    return labels

# read all the training data
training_data = [read_data(x) for x in train_files]
training_labels = [read_label(x) for x in train_files]
# flatten the list of lists
training_data = [item for sublist in training_data for item in sublist]
training_labels = [item for sublist in training_labels for item in sublist]
print(f'# of training data: {len(training_data)}')
print(f'# of training labels: {len(training_labels)}')

# read all the testing data
test_data = [read_data(x) for x in test_files]
# flatten the list of lists
test_data = [item for sublist in test_data for item in sublist]
test_labels = [read_label(x) for x in test_files]
test_labels = [item for sublist in test_labels for item in sublist]
print(f'# of testing data: {len(test_data)}')
print(f'# of testing labels: {len(test_labels)}')

class sgHCM:
    # first, initial the class with k value
    # then, call the class with test sample and train data
    def __init__(self, k=3, max_iter=10, tol=1e-5):
        # number of clusters
        self.k = k
        # maximum iteration
        self.max_iter = max_iter
        # tolerance (epsilon)
        self.tol = tol
        # initialize the list of distances between old and new centroids very high
        self.Et = np.inf
        self.train_data = None
        self.centroids = None

    def __call__(self, test_sample):
        # compute the distance between test_sample and the centroids
        self.distances = self.compute_distance(test_sample, self.centroids)

        # find the index of the closest centroid
        closest_centroid = np.argmin(self.distances)

        # assign the test sample to the closest centroid
        self.predicted_label = closest_centroid

        return self.predicted_label, self.centroids[closest_centroid]
        
    def fit(self, train_data):
        self.train_data = train_data
        # randomly select k centroids from train_data (or prototype vectors)
        self.centroids = np.random.choice(train_data, size=self.k, replace=False)

        # initialize the list of clusters
        self.clusters = [[] for _ in range(self.k)]

        # initialize the list of old centroids
        self.old_centroids = None

        # initialize the list of new centroids
        self.new_centroids = None

        # initialize the list of distances between test sample and each train data
        self.distances = None

        # initialize the list of predicted labels
        self.predicted_label = None

        self.u_ik = None
        self.n_it = None

        for t in range(self.max_iter):
            print(f'\nIteration {t + 1}')
            # assign each train data to the closest centroid
            self.clusters, self.u_ik = self.assign_clusters(self.train_data)

            # n_it is the number of train samples in the i-th cluster
            self.n_it = np.sum(self.u_ik, axis=0)

            # update the centroids
            self.old_centroids = self.centroids
            self.new_centroids = self.update_centroids(self.train_data)
            
            _ = self.centroids_uniqueness(self.new_centroids)
            # print(f'New centroids: {self.new_centroids}')

            # compute the difference between old and new centroids
            self.Et = self.compute_terminal_measure(self.old_centroids, self.new_centroids)

            # if the difference is less than the tolerance, stop the iteration
            if self.Et < self.tol:
                break

            # otherwise, update the centroids and continue
            self.centroids = self.new_centroids

        print(f'Final centroids: {self.centroids}')
        # return the list of clusters
        return self.clusters

    def assign_clusters(self, train_data):
        # compute the distance between the train data and each centroid
        # initialize the list of clusters
        clusters = [[] for _ in range(self.k)]

        # for each train data
        for i, train_sample in enumerate(train_data):
            # compute the distance between test_sample and the centroid
            self.distances = self.compute_distance(train_sample, self.centroids)

            # find the index of the closest centroid
            closest_centroid = np.argmin(self.distances)

            # assign the train sample to the closest centroid
            clusters[closest_centroid].append(i)

        # create the u_ik matrix, which is a matrix of 0s and 1s
        # u_ik[i][j] = 1 if the i-th train sample belongs to the j-th cluster
        # u_ik[i][j] = 0 otherwise
        u_ik = np.zeros((len(train_data), self.k))
        for i, cluster in enumerate(clusters):
            for j in cluster:
                u_ik[j][i] = 1

        self.n_it = np.sum(u_ik, axis=0)

        return clusters, u_ik
    
    def update_centroids(self, train_data):
        # compute the distance between each sample and other samples in the same cluster
        # the sample with the minimum sum of the distances to other samples in the same cluster is the new centroid for that cluster  

        # initialize the list of new centroids
        new_centroids = []

        # for each cluster
        for i in range(len(self.clusters)):
            # c_ij is a list of the sum of the distances to other samples in the same cluster
            c_ij = []

            for j in range(len(self.clusters[i])):
                # compute the distance between the sample and other samples in the same cluster excluding itself
                # list of samples excluding the sample itself
                samples = [x for x in self.clusters[i] if x != j]

                # take train_data at the index of samples
                train = np.array(train_data.copy())
                other_samples = train[samples]

                distances = self.compute_distance(train_data[j], other_samples)
                
                c_ij.append(np.sum(distances/self.n_it[i], axis=0))

            # find the index of the sample with the minimum sum 
            # of the distances to other samples in the same cluster
            try:
                alpha_q = np.argmin(c_ij)
            # except there is an error, which is the case when c_ij is empty, redraw the centroids
            except:
                print(f'Found empty cluster')
                self.centroids = np.random.choice(self.train_data, size=self.k, replace=False)
                return self.centroids

            # add the new centroid to the list
            new_centroids.append(train_data[alpha_q])

        return new_centroids
    
    def centroids_uniqueness(self, centroids):
        # check if the new centroids are unique
        # return True if the new centroids are unique
        # return False if the new centroids are not unique
        # print(f'Checking centroids uniqueness...')
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                if i != j:
                    if centroids[i] == centroids[j]:
                        print(f'Found centroids not unique')
                        # redraw the centroids
                        self.centroids = np.random.choice(self.train_data, size=self.k, replace=False)
                        return False
        return True
    
    def levenshtein_distance(self, s1, s2):
        # Create a matrix to store the distances between substrings of s1 and s2
        distance_matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

        # Initialize the first row and column of the matrix
        for i in range(len(s1) + 1):
            distance_matrix[i][0] = i
        for j in range(len(s2) + 1):
            distance_matrix[0][j] = j

        # Fill in the matrix using dynamic programming
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j] + 1,  # Deletion
                    distance_matrix[i][j - 1] + 1,  # Insertion
                    distance_matrix[i - 1][j - 1] + cost  # Substitution
                )

        # The final value in the bottom-right corner of the matrix is the Levenshtein distance
        return distance_matrix[len(s1)][len(s2)]

    def compute_distance(self, test_sample, train_data):
        # compute the distance between test_sample and each train data
        # return a list of distances
        distances = []
        for train_sample in train_data:
            distance = self.levenshtein_distance(test_sample, train_sample)
            distances.append(distance)
        return distances
    
    def compute_terminal_measure(self, old_V, new_V):
        # compute the levenstein distance between old and new centroids
        # return the sum of the levenstein distances
        Et = []
        for i in range(len(old_V)):
            Et.append(self.levenshtein_distance(old_V[i], new_V[i]))
    
        Et = np.array(Et)
        Et = np.sum(Et)
        print(f'Terminal measure: Et = {Et}')

        # return the tolerance
        return Et
    
# perform 10-fold cross validation on sgHCM classifier
def cross_validation(train_data, test_data, k_num = 2, fold=10):
    # shuffle the number of samples in the training data 
    # and rearrange the labels accordingly

    # combine the training data and labels
    train_data = np.array(train_data)
    train_data = np.c_[train_data, training_labels]

    # shuffle the training data
    np.random.shuffle(train_data)

    # split the training data and labels
    training_labels = train_data[:, -1]
    train_data = train_data[:, :-1]    

    fold_size = len(train_data) // fold

    for i in range(fold):
        # take one fold as training data
        data_train_fold = train_data[i * fold_size: (i + 1) * fold_size]
        data_test_fold = test_data[i * fold_size: (i + 1) * fold_size]

        # show number of unique labels in the training data
        print(f'\nNumber of unique labels in the training data: {len(np.unique(training_labels))}')

        num_labels = len(np.unique(training_labels))
        ############################ sgHCM classifier ##########################
        print(f"\nFold {i+1}")
        classifier = sgHCM(k=num_labels)
        classifier.fit(data_train_fold)

        # print('\nPredicting the test dataset...')
        print(f'Predictions: (test sample, predicted label, predicted centroid)')
        predictions = []
        # test the first 5 samples
        for test_sample in data_test_fold[:5]:
            predicted_label, predicted_centroids = classifier(test_sample)
            print(f'{test_sample}, {predicted_label}, {predicted_centroids}')
            predictions.append([test_sample, predicted_label, predicted_centroids])
     
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

cross_validation(training_data, test_data)
