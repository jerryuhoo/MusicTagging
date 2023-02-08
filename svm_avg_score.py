mfcc_n = 25
score = "list_file_" + str(mfcc_n) + ".txt"
accuracy = []
f1 = []
with open(score, "r") as f:
    for line in f:
        row = [x for x in line[1:-2].split(',')]
        accuracy.append(float(row[1]))
        f1.append(float(row[2]))
mean_accuracy = sum(accuracy) / len(accuracy)
mean_f1 = sum(f1) / len(f1)
# print(accuracy)
print("mfcc " + str(mfcc_n) + " mean_accuracy:" + \
    str(mean_accuracy) + " mean_f1:" + str(mean_f1))
