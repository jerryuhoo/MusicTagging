import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import h5py

# hyper parameters
n_components = 25
plot_graph = False

with h5py.File("preprocessed/10/training/mfcc.h5", "r") as f:
    X_train = f["mfcc"][:]

with h5py.File("preprocessed/10/training/label.h5", "r") as f:
    y_train = f["label"][:]

with h5py.File("preprocessed/10/testing/mfcc.h5", "r") as f:
    X_test = f["mfcc"][:]

with h5py.File("preprocessed/10/testing/label.h5", "r") as f:
    y_test = f["label"][:]

X_train = np.array(X_train)
X_train = np.mean(X_train, axis=-1)
y_train = np.array(y_train)

X_test = np.array(X_test)
X_test = np.mean(X_test, axis=-1)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

# load csv data to get column names
csv_dir = "../data/magnatagatune/annotations_final_new.csv"
df = pd.read_csv(csv_dir, sep="\t")
label_names = df.columns.values
n_cols = df.shape[1] - 2

# loop through all labels
results = []

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# pca = PCA(n_components=n_components)
# X = pca.fit_transform(X)
# print("after PCA", X.shape)

for label_index in range(0, n_cols):
    print("label_index", label_index)
    y_train_one_label = y_train[:, label_index]
    y_test_one_label = y_test[:, label_index]

    current_label = label_names[label_index]
    print("now training:", current_label)

    # Initialize the sampler
    print("under sample:")
    sampler = RandomUnderSampler(sampling_strategy="majority")

    # Fit and apply the sampler to your data
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train_one_label)

    # X_test_resampled, y_test_one_label_resampled = sampler.fit_resample(
    #     X_test, y_test_one_label
    # )

    print("Number of positive labels in training data:", sum(y_train_one_label == 1))
    print("Number of negative labels in training data:", sum(y_train_one_label == 0))
    print("Number of positive labels in resampled data:", sum(y_resampled == 1))
    print("Number of negative labels in resampled data:", sum(y_resampled == 0))

    # Train the SVM model
    clf = svm.SVC(kernel="linear", C=1, random_state=42, verbose=True)
    clf.fit(X_resampled, y_resampled)

    # Evaluate the model on the test set
    accuracy = clf.score(X_test, y_test_one_label)
    # accuracy = clf.score(X_test_resampled, y_test_one_label_resampled)
    print("Accuracy:", accuracy)

    # Predict the labels for the test data using the trained model
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test_one_label, y_pred)
    # y_pred = clf.predict(X_test_resampled)
    # f1 = f1_score(y_test_one_label_resampled, y_pred)

    # Print the F1-score
    print("F1-score:", f1)

    model_dir = "models/svm"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model to a file
    joblib.dump(clf, os.path.join("models", "svm", "svm_" + current_label + ".joblib"))
    print("svm model saved!")

    # Load the model from a file
    # clf_loaded = joblib.load(
    #     os.path.join("models", "svm", "svm_" + current_label + ".joblib")
    # )

    results.append([current_label, accuracy, f1])

    if plot_graph:
        # Plot the data and the decision boundary
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis", s=2)

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        ax.contour(
            XX,
            YY,
            Z,
            colors="red",
            levels=[-1, 0, 1],
            alpha=0.5,
            linestyles=["--", "-", "--"],
        )
        # ax.scatter(
        #     clf.support_vectors_[:, 0],
        #     clf.support_vectors_[:, 1],
        #     s=5,
        #     linewidth=1,
        #     facecolors="none",
        #     edgecolors="k",
        # )

        plt.savefig("svm_" + current_label + ".png", dpi=100)
        # clear the plot
        plt.clf()
        print(current_label + " finished!")

# extract the accuracy and f1 scores into separate lists
accuracies = [x[1] for x in results]
f1_scores = [x[2] for x in results]

# compute the mean accuracy and f1 score using numpy.mean()
mean_accuracy = np.mean(accuracies)
mean_f1 = np.mean(f1_scores)

average = ["average", mean_accuracy, mean_f1]

with open("list_file_" + str(n_components) + ".txt", "w") as file:
    for row in results:
        print(row)
        file.write(str(row) + "\n")
    file.write(str(average) + "\n")

# print the results
print("Mean accuracy:", mean_accuracy)
print("Mean F1 score:", mean_f1)
print("Done!")
