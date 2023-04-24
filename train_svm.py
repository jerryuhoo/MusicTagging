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

from imblearn.under_sampling import RandomUnderSampler
import h5py

# hyper parameters
n_components = 25
plot_graph = False

with h5py.File("preprocessed/10/training/mfcc.h5", "r") as f:
    X = f["mfcc"][:]

with h5py.File("preprocessed/10/training/label.h5", "r") as f:
    y = f["label"][:]

X = np.array(X)
X = np.mean(X, axis=-1)
y_all = np.array(y)
print(X.shape)
print(y_all.shape)

assert X.shape[0] == y_all.shape[0]

# load csv data to get column names
csv_dir = "../data/magnatagatune/annotations_final_new.csv"
df = pd.read_csv(csv_dir, sep="\t")
label_names = df.columns.values
n_cols = df.shape[1] - 2

# loop through all labels
results = []
for label_index in range(0, n_cols):
    y = y_all[:, label_index]

    current_label = label_names[label_index]
    print("now training:", current_label)
    # print("before PCA", X.shape)

    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    # print("after PCA", X.shape)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    # print(scaler.mean_)  # Prints the mean of each feature
    # print(scaler.var_)  # Prints the variance of each feature
    # np.save("preprocessed/mean.npy", X_mean)
    # np.save("preprocessed/var.npy", X_var)
    # print(X_normalized)

    # Initialize the sampler
    print("under sample:")
    sampler = RandomUnderSampler(sampling_strategy="majority")

    # Fit and apply the sampler to your data
    X_resampled, y_resampled = sampler.fit_resample(X_normalized, y)

    # Split the data into training and test sets
    print("split:")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    print("Number of positive labels in training data:", sum(y_train == 1))
    print("Number of negative labels in training data:", sum(y_train == 0))
    print("Number of positive labels in testing data:", sum(y_test == 1))
    print("Number of negative labels in testing data:", sum(y_test == 0))

    # Train the SVM model
    clf = svm.SVC(kernel="linear", C=1, random_state=42, verbose=True)
    clf.fit(X_train, y_train)

    # Evaluate the model on the test set
    # X_test_normalized = scaler.transform(X_test)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)

    # Predict the labels for the test data using the trained model
    y_pred = clf.predict(X_test)
    # print("y_pred", y_pred)
    # Calculate the F1-score for the predictions
    f1 = f1_score(y_test, y_pred)

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


with open("list_file_" + str(n_components) + ".txt", "w") as file:
    for row in results:
        print(row)
        file.write(str(row) + "\n")
