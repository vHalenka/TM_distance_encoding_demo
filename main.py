from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time

# Hyperparameter optimization
import optuna

# Plot Drawing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
# Turn off annoying matplotlib log messages
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt


bits_total = 6
clauses = 4

orig_data_train = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0], # Class 0
							[0, 0, 1, 0, 1, 0, 0, 0, 0, 0], # "101" anywhere
							[0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
							[1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 1],

							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Class 1
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # all zeros
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

							[1, 1, 0, 0, 0, 1, 0, 0, 1, 1], # Class 2
							[0, 0, 1, 0, 0, 0, 1, 1, 0, 0], # random noise
							[0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
							[0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
							[0, 1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.uint32)


X_train = np.zeros((10*15, 6), dtype=np.uint32)
Y_train = np.array([0]*50 + [1]*50 + [2]*50, dtype=np.uint32)

orig_data_test = np.array([ [1, 0, 1, 0, 0, 0, 0, 0, 0, 0], # Class 0
							[0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 1],

							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Class 1
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

							[1, 0, 0, 0, 1, 0, 0, 0, 1, 1], # Class 2
							[0, 0, 1, 0, 0, 0, 0, 1, 1, 0], # just two adjecent ones "11"
							[0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
							[0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
							[0, 1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.uint32)

X_test = np.zeros((10*15, 6), dtype=np.uint32)
Y_test = np.array([0]*50 + [1]*50 + [2]*50, dtype=np.uint32)

for i, X in enumerate(orig_data_test):
	for bit in range(10):
		window = np.zeros(6, dtype=np.uint32)
		window[0] = X[bit] # store the value

		# Find the next bit of the same kind
		if X[bit] == 1:
			for next_bit_i in range(bit+1, 10):
				if X[next_bit_i] == 1:
					next_bit_therm = np.zeros(5, dtype=np.uint32)

					# Relative distance
					next_bit_aux = next_bit_i-bit
					# Thermometer encode
					for z in range(5):
						next_bit_therm[z] = next_bit_aux >= z + 1

					# Store the thermometer code
					window[1:6] = next_bit_therm
					break

		X_test[i*10+bit] = window # [X[bit], next_bit_therm[0], next_bit_therm[1], next_bit_therm[2], next_bit_therm[3], next_bit_therm[4]]

for i, X in enumerate(orig_data_train):
	for bit in range(10):
		window = np.zeros(6, dtype=np.uint32)
		window[0] = X[bit] # store the value

		# Find the next bit of the same kind
		if X[bit] == 1:
			for next_bit_i in range(bit+1, 10):
				if X[next_bit_i] == 1:
					next_bit_therm = np.zeros(5, dtype=np.uint32)

					# Relative distance
					next_bit_aux = next_bit_i-bit
					# Thermometer encode
					for z in range(5):
						next_bit_therm[z] = next_bit_aux >= z + 1

					# Store the thermometer code
					window[1:6] = next_bit_therm
					break

		X_train[i*10+bit] = window # [X[bit], next_bit_therm[0], next_bit_therm[1], next_bit_therm[2], next_bit_therm[3], next_bit_therm[4]]
print(X_test[120:130])


X_train = X_train.reshape(15, 1, 10, 6)
X_test = X_test.reshape(15, 1, 10, 6)
Y_test  = np.array([0]*5 + [1]*5 + [2]*5, dtype=np.uint32)
Y_train = np.array([0]*5 + [1]*5 + [2]*5, dtype=np.uint32)

print("X_train", X_train.shape)
print("Y_train",Y_test.shape)

print("X_test",X_test.shape)
print("Y_test",Y_test.shape)


################################################################################
## Dataset generation
""" number_of_features = 10
number_of_distance_bits = 2

bits_total = number_of_features + number_of_distance_bits + direction_bits
# Generate the training and test data
X_train = np.zeros((500,bits_total),dtype=np.uint32) # 500 samples, 1000 features
k = 3  # Set distance between the two numbers
Y_train = np.zeros(500, dtype=np.uint32)  # 500 labels
def generate_data(X, Y, number_of_features, bits_total, k):
	for i in range(number_of_features):
		class_label = i % 3  # Assign class labels cyclically
		if class_label == 1:
			pos1 = np.random.randint(0, X.shape[1] - k)
			pos2 = pos1 + k
			X[i, pos1] = 1
			X[i, pos2] = 1
		elif class_label == 2:
			pos1 = np.random.randint(0, X.shape[1])
			pos2 = np.random.randint(0, X.shape[1])
			X[i, pos1] = 1
			X[i, pos2] = 1
		Y[i] = class_label

number_of_features = 10
bits_total = number_of_features + number_of_distance_bits + direction_bits
k = 3

X_train = np.zeros((500, bits_total), dtype=np.uint32)
Y_train = np.zeros(500, dtype=np.uint32)
generate_data(X_train, Y_train, number_of_features, bits_total, k)

X_test = np.zeros((500, bits_total), dtype=np.uint32)
Y_test = np.zeros(500, dtype=np.uint32)
generate_data(X_test, Y_test, number_of_features, bits_total, k)
 """
################################################################################
## Optuna hyperparameter optimization
""" def objective(trial):
	s = trial.suggest_int('s', 10, 100)
	T = trial.suggest_float('T', 0.1, 2.0)

	tm = TMClassifier(clauses, s, T, patch_dim=(1,1), platform='CPU', weighted_clauses=True)

	for i in range(100):
		tm.fit(X_train, Y_train)

	accuracy = (tm.predict(X_test) == Y_test).mean()
	return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_accuracy = study.best_value

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)
#tm = TMClassifier(clauses, best_params[0], best_params[1], platform='CPU')
#tm = TMClassifier(clauses, 58, 1.5, patch_dim=[1], platform='CPU', weighted_clauses=True)
 """
tm = TMClassifier(clauses, T=int((np.sqrt(clauses)/2 + 2)*10), s=1.533, patch_dim=(1,1), platform='CPU', weighted_clauses=True)
for i in range(100):
 		tm.fit(X_train, Y_train)



np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

print("\nClass 0 Positive Clauses:\n")

precision = tm.clause_precision(0, 0, X_test, Y_test)
recall = tm.clause_recall(0, 0, X_test, Y_test)

for j in range(clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(bits_total*2):
		if tm.get_ta_action(j, k, the_class = 0, polarity = 0):
			if k < bits_total:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-bits_total))
	print(" ∧ ".join(l))

print("\nClass 0 Negative Clauses:\n")

precision = tm.clause_precision(0, 1, X_test, Y_test)
recall = tm.clause_recall(0, 1, X_test, Y_test)

for j in range(clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(bits_total*2):
		if tm.get_ta_action(j, k, the_class = 0, polarity = 1):
			if k < bits_total:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-bits_total))
	print(" ∧ ".join(l))

print("\nClass 1 Positive Clauses:\n")

precision = tm.clause_precision(1, 0, X_test, Y_test)
recall = tm.clause_recall(1, 0, X_test, Y_test)

for j in range(clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(bits_total*2):
		if tm.get_ta_action(j, k, the_class = 1, polarity = 0):
			if k < bits_total:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-bits_total))
	print(" ∧ ".join(l))

print("\nClass 1 Negative Clauses:\n")

precision = tm.clause_precision(1, 1, X_test, Y_test)
recall = tm.clause_recall(1, 1, X_test, Y_test)

for j in range(clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(bits_total*2):
		if tm.get_ta_action(j, k, the_class = 1, polarity = 1):
			if k < bits_total:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-bits_total))
	print(" ∧ ".join(l))
 
print("\nClass 2 Positive Clauses:\n")

precision = tm.clause_precision(2, 0, X_test, Y_test)
recall = tm.clause_recall(2, 0, X_test, Y_test)

for j in range(clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(2, 0, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(bits_total*2):
		if tm.get_ta_action(j, k, the_class = 2, polarity = 0):
			if k < bits_total:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-bits_total))
	print(" ∧ ".join(l))

print("\nClass 2 Negative Clauses:\n")

precision = tm.clause_precision(2, 1, X_test, Y_test)
recall = tm.clause_recall(2, 1, X_test, Y_test)

for j in range(clauses//2):
	print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(2, 1, j), precision[j], recall[j]), end=' ')
	l = []
	for k in range(bits_total*2):
		if tm.get_ta_action(j, k, the_class = 2, polarity = 1):
			if k < bits_total:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-bits_total))
	print(" ∧ ".join(l))

print("\nClause Co-Occurence Matrix:\n")
print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

print("\nLiteral Frequency:\n")
print(tm.literal_clause_frequency())

print("\nNumber of Literals:\n")
print(len(tm.literal_clause_frequency()))

accuracy = (tm.predict(X_test) == Y_test).mean()
print("Accuracy:", accuracy*100, "%")

#Confusion Matrix TM 1
conf_matrix1 = confusion_matrix(Y_test, tm.predict(X_test))
# Calculate percentages
conf_matrix_percent1 = conf_matrix1 / conf_matrix1.sum(axis=1, keepdims=True) * 100
conf_matrix_percent1 = np.nan_to_num(conf_matrix_percent1, 0)  # Replace NaNs with 0

# Plot the confusion matrix with percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent1, annot=True, fmt=".1f", cmap="Blues")
plt.title("Confusion Matrix run 1 (Percentages)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()