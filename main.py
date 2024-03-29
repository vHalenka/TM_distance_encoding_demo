from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time
import optuna
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt #drawing plots
import seaborn as sns

import logging
import pickle
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
################################################################################


# TM parameters:
clauses = 2
epochs = 500


# Dataset:
number_of_samples = 50
number_of_values = 100
number_of_value_bits = 2
number_of_distance_bits = 4
k = 3

position_bits = number_of_values

patch_dim = (1,1)
bits_total = position_bits + number_of_value_bits + number_of_distance_bits


################################################################################
## Dataset generation

# Generate the training and test data
def generate_data(X, Y, k):
	for i in range(X.shape[0]):
		class_label = i % 3  # Assign class labels cyclically
		if class_label == 1:
			pos1 = np.random.randint(0, X.shape[1] - k)
			pos2 = pos1 + k
			X[i, pos1] = 1
			X[i, pos2] = 1
		elif class_label == 2:
			pos1 = np.random.randint(0, X.shape[1])
			pos2 = np.random.randint(0, X.shape[1])
			while pos2 == pos1 or pos2 == pos1 + k or pos2 == pos1 - k:
				pos2 = np.random.randint(0, X.shape[1])
			X[i, pos1] = 1
			X[i, pos2] = 1
		Y[i] = class_label

X_train_org = np.zeros((number_of_samples, number_of_values), dtype=np.uint32)
Y_train = np.zeros(number_of_samples, dtype=np.uint32)
generate_data(X_train_org, Y_train, k)

X_test_org = np.zeros((number_of_samples, number_of_values), dtype=np.uint32)
Y_test = np.zeros(number_of_samples, dtype=np.uint32)
generate_data(X_test_org, Y_test, k)

print("Original data:")
print(X_train_org)
print(X_test_org)

################################################################################

def encode_to_therm(X, n_value_bits, n_distance_bits): #X in samples, 1, values, values+distance)
	X_with_dist = np.zeros((X.shape[0], 1, X.shape[1] , n_value_bits+n_distance_bits), dtype=np.uint32)
	for sample in range(X.shape[0]):
		for i, value in enumerate(X[sample]):
			window = np.zeros(n_value_bits+n_distance_bits, dtype=np.uint32)

   			# store the value in thermometer code
			window[0:n_value_bits] = (value >= np.arange(1, n_value_bits+1)).astype(np.uint32)  

			if value == 1:
				for j in range(i+1, X.shape[1]):
					if X[sample][j] == 1:
						next_bit_therm = np.zeros(n_distance_bits, dtype=np.uint32)

						# Relative distance
						next_bit_aux = j - i
						# Thermometer encode
						#next_bit_therm = (next_bit_aux >= np.arange(1, n_distance_bits+1)).astype(np.uint32)

						# Store the thermometer code after the value
						window[n_value_bits:] = next_bit_therm
						break

			X_with_dist[sample, 0, i, :] = window
	return X_with_dist

X_train = encode_to_therm(X_train_org, number_of_value_bits, number_of_distance_bits)
X_test = encode_to_therm(X_test_org, number_of_value_bits, number_of_distance_bits)
print("Data with distance (val bits:", number_of_value_bits, "dist bits:", number_of_distance_bits, "):")
print(X_train)
print(X_test)

def thermometer_to_integer(thermometer_bits, lengths, include_negatives=False):
	"""
	thermometer_bits: list of bits in thermometer code (X_train (samples, 1, values, value_bits+distance_bits))
	lengths: number of bits for each thermometer code (values, distance)
	"""
	if include_negatives:
		neg = 1
		lengths = lengths + lengths
	else:
		neg = 0
	result_shape = thermometer_bits.shape[:-1] + (len(lengths),)
	result = np.zeros(result_shape, dtype=np.uint32)
	for sample_i in np.ndindex(thermometer_bits.shape[:-1]):
		therm_pointer = 0 
		for i, length in enumerate(lengths):
			therm_bits_focus = np.array(thermometer_bits[tuple(sample_i) + (slice(therm_pointer, therm_pointer+length),)], dtype=np.uint32)
			if neg and i>=len(lengths)//2:
				value = length
				bits = range(length)
			else:
				value = 0
				bits = reversed(range(length))
			for bit in bits:
				if therm_bits_focus[bit]:
					value = bit
				else:
					break
			result[tuple(sample_i) + (i,)] = value
			therm_pointer += length

	return result


print("Thermometer to integer, orig data")
print(thermometer_to_integer(X_train, [number_of_value_bits, number_of_distance_bits]))
print(thermometer_to_integer(X_test, [number_of_value_bits, number_of_distance_bits]))

print("X_train", X_train.shape)
print("Y_train",Y_test.shape)

print("X_test",X_test.shape)
print("Y_test",Y_test.shape)


################################################################################
## Optuna hyperparameter optimization
""" def objective(trial):
	s = trial.suggest_int('s', 10, 100)
	T = trial.suggest_float('T', 0.1, 2.0)

	tm = TMClassifier(clauses, s, T, patch_dim=patch_dim, platform='CPU', weighted_clauses=True)

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
#tm = TMClassifier(clauses, 58, 1.5, patch_dim=patch_dim, platform='CPU', weighted_clauses=True) """
 
print("Training TM1 with clauses:", clauses)
print(X_train.shape, X_train, file=open("X_train.txt", "w"))
print(X_test.shape, X_test, file=open("X_test.txt", "w"))
tm = TMClassifier(clauses, T=int((np.sqrt(clauses)/2 + 2)*10), s=13, patch_dim=patch_dim, platform='CPU', weighted_clauses=True)
for i in range(epochs):
		tm.fit(X_train, Y_train)
		if i < epochs//2:
			accuracy = (tm.predict(X_test) == Y_test).mean()
		if accuracy == 1:
			break

################################################################################
# Read and Store the clauses

all_clauses = np.array([], dtype=np.uint64)  # class, clause
k = 0  # ID
for Class in range(tm.number_of_classes):
	class_patches = np.array([], dtype=np.uint64)
	for Clause in range(tm.number_of_clauses):
		block = [tm.get_ta_action(Clause, bit, Class) for bit in np.arange(2 * bits_total)]
		class_patches = np.append(class_patches, block)
	all_clauses = np.append(all_clauses, class_patches)

all_clauses = all_clauses.reshape(tm.number_of_classes, tm.number_of_clauses, 2 * bits_total)


# Store the clauses using pickle
with open("clauses.pkl", "wb") as f:
	pickle.dump(all_clauses, f)

# Load the clauses using pickle
with open("clauses.pkl", "rb") as f:
	all_clauses = pickle.load(f)
print(all_clauses)
print(all_clauses.shape)

################################################################################
## Decode the patches for searching

#def decode_patches(patches, n_value_bits, n_distance_bits):

all_clauses = thermometer_to_integer(all_clauses, [position_bits, number_of_value_bits, number_of_distance_bits], include_negatives=True)

print("Thermometer to integer from clauses")
print("value, distance, negated value, negated distance)")
print(all_clauses)
print("all_clauses.shape (class, clause, literal+neg_literal):", all_clauses.shape)

################################################################################
## Examine cooccurance

cc_matrix_org = tm.clause_co_occurrence(X_train)
cc_matrix = cc_matrix_org.toarray()  # Convert sparse matrix to dense

cc_diagonal = np.diagonal(cc_matrix)
np.fill_diagonal(cc_matrix, 0) # Set to zero so it doesnt target itself

print("")
print("cc_matrix:")
print(cc_matrix.shape)
print(cc_matrix)

################################################################################
## Assert score instead of co_occurance

# Initialize score matrix
score_matrix = np.zeros_like(cc_matrix, dtype=float)

# Calculate total count
total_count = np.sum(cc_matrix)

# Calculate probabilities and scores
for i in range(cc_matrix.shape[0]):
    for j in range(cc_matrix.shape[1]):
        p_c1 = np.sum(cc_matrix[i, :]) / total_count
        p_c2 = np.sum(cc_matrix[:, j]) / total_count
        p_c1_c2 = cc_matrix[i, j] / total_count

        # Calculate score and store in score_matrix
        if p_c1_c2 > 0:  # To avoid division by zero and log of zero
            score_matrix[i, j] = np.log(p_c1_c2 / (p_c1 * p_c2))

print(score_matrix)


# For each ID get the highest co-occuring ID
cc_best_matches = np.argsort(score_matrix)
################################################################################
## Find the best match in original data
def encode_to_therm_with_dist(X, n_value_bits, n_distance_bits): #X in samples, 1, values, values+distance

	X_with_dist = np.zeros((X.shape[0], 1, X.shape[1] , n_value_bits+n_distance_bits), dtype=np.uint32)
 
	# store the value
	for sample in range(X.shape[0]):
		for i, value in enumerate(X[sample]):
   			# store the value in thermometer code
			X_with_dist[sample, 0, i, :n_value_bits] = (value >= np.arange(1, n_value_bits+1)).astype(np.uint32)  
			
   # store the distance
	f = open('search_log.txt', 'w')
	print("\n\nLooking for best match in original data\n", file=f)
	for Class_i, Class in enumerate(all_clauses):
		for clause_i, clause in enumerate(Class):
			ID_start = (Class_i*tm.number_of_clauses) + clause_i
			print("ID:", ID_start, "Looking for value between", clause[1], clause[4], file=f)
			for sample in X_test_org:
				for value_i, value in enumerate(sample):
					if (value == clause[1]) & (value <= clause[4]):
						print("ID:", ID_start, "Found value:", value, "at position:", value_i, "at sample:", sample, file=f)
						best_matches_id = cc_best_matches[ID_start]
						ID_target = best_matches_id[0] # select first best match
						target_clause = all_clauses[ID_target//tm.number_of_clauses][ID_target%tm.number_of_clauses]
						# search for the best match in the original data
						print("ID:", ID_start, "Looking for highest co_ocur match, value between", target_clause[1], target_clause[4], "with ID:", ID_target, file=f)
						target_value_i = value_i + 1
						if target_value_i < len(sample):  # Check if target_value_i is within the valid range
							target_value = sample[target_value_i]
							if (target_value == target_clause[1]) & (target_value <= target_clause[4]):
								print("ID:", ID_start, "Match found, target ID:", ID_target, "at target position:", target_value_i, "and target value:", target_value, file=f)
								
								# store the value in thermometer code
								target_therm = (target_value_i-value_i >= np.arange(1, n_distance_bits+1)).astype(np.uint32)
								# Store the thermometer code after the value
								X_with_dist[sample, 0, value_i, n_value_bits:] = target_therm
								print("Search ended successfully", file=f)
								break
	return X_with_dist

X_train2 = encode_to_therm_with_dist(X_train_org, number_of_value_bits, number_of_distance_bits)
X_test2  = encode_to_therm_with_dist(X_test_org,  number_of_value_bits, number_of_distance_bits)

################################################################################
## Train second tm with the new data
print("Training TM2 with clauses:", clauses)
tm2 = TMClassifier(clauses, T=int((np.sqrt(clauses)/2 + 2)*10), s=1.533, patch_dim=patch_dim, platform='CPU', weighted_clauses=True)
for i in range(epochs):
		tm2.fit(X_train2, Y_train)
		if i < epochs//2:
			accuracy = (tm2.predict(X_test2) == Y_test).mean()
		if accuracy == 1:
			break

################################################################################
# Printouts TM 1
np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

for class_num in range(tm.number_of_classes):
	print(f"\nTM1 Class {class_num} Positive Clauses:\n")
	precision = tm.clause_precision(class_num, 0, X_test, Y_test)
	recall = tm.clause_recall(class_num, 0, X_test, Y_test)
	for j in range(clauses//2):
		print(f"Clause #{j} W:{tm.get_weight(class_num, 0, j)} P:{precision[j]:.2f} R:{recall[j]:.2f} ", end=' ')
		l = []
		for k in range(bits_total*2):
			if tm.get_ta_action(j, k, the_class=class_num, polarity=0):
				if k < bits_total:
					l.append(f" x{k}")
				else:
					l.append(f"¬x{k-bits_total}")
		print(" ∧ ".join(l))

	print(f"\nTM1 Class {class_num} Negative Clauses:\n")
	precision = tm.clause_precision(class_num, 1, X_test, Y_test)
	recall = tm.clause_recall(class_num, 1, X_test, Y_test)
	for j in range(clauses//2):
		print(f"Clause #{j} W:{tm.get_weight(class_num, 1, j)} P:{precision[j]:.2f} R:{recall[j]:.2f} ", end=' ')
		l = []
		for k in range(bits_total*2):
			if tm.get_ta_action(j, k, the_class=class_num, polarity=1):
				if k < bits_total:
					l.append(f" x{k}")
				else:
					l.append(f"¬x{k-bits_total}")
		print(" ∧ ".join(l))


print("\nClause Co-Occurence Matrix:\n")
print(tm.clause_co_occurrence(X_test, percentage=True).toarray())

print("\nLiteral Frequency:\n")
print(tm.literal_clause_frequency())

print("\nNumber of Literals:\n")
print(len(tm.literal_clause_frequency()))

accuracy_tm = (tm.predict(X_test) == Y_test).mean()
accuracy_tm2 = (tm2.predict(X_test2) == Y_test).mean()

print("Accuracy for tm:", accuracy_tm*100, "%")
print("Accuracy for tm2:", accuracy_tm2*100, "%")


################################################################################
# Printouts TM 2

for class_num in range(tm2.number_of_classes):
	print(f"\nTM2 Class {class_num} Positive Clauses:\n")
	precision = tm2.clause_precision(class_num, 0, X_test, Y_test)
	recall = tm2.clause_recall(class_num, 0, X_test, Y_test)
	for j in range(clauses//2):
		print(f"Clause #{j} W:{tm2.get_weight(class_num, 0, j)} P:{precision[j]:.2f} R:{recall[j]:.2f} ", end=' ')
		l = []
		for k in range(bits_total*2):
			if tm2.get_ta_action(j, k, the_class=class_num, polarity=0):
				if k < bits_total:
					l.append(f" x{k}")
				else:
					l.append(f"¬x{k-bits_total}")
		print(" ∧ ".join(l))

	print(f"\nTM2 Class {class_num} Negative Clauses:\n")
	precision = tm2.clause_precision(class_num, 1, X_test, Y_test)
	recall = tm2.clause_recall(class_num, 1, X_test, Y_test)
	for j in range(clauses//2):
		print(f"Clause #{j} W:{tm2.get_weight(class_num, 1, j)} P:{precision[j]:.2f} R:{recall[j]:.2f} ", end=' ')
		l = []
		for k in range(bits_total*2):
			if tm2.get_ta_action(j, k, the_class=class_num, polarity=1):
				if k < bits_total:
					l.append(f" x{k}")
				else:
					l.append(f"¬x{k-bits_total}")
		print(" ∧ ".join(l))


print("\nClause Co-Occurence Matrix:\n")
print(tm2.clause_co_occurrence(X_test, percentage=True).toarray())

print("\nLiteral Frequency:\n")
print(tm2.literal_clause_frequency())

print("\nNumber of Literals:\n")
print(len(tm2.literal_clause_frequency()))

print("Accuracy for tm:", accuracy_tm*100, "%")
print("Accuracy for tm2:", accuracy_tm2*100, "%")
################################################################################
# Confusion Matrix TM 1
conf_matrix1 = confusion_matrix(Y_test, tm.predict(X_test))
# Calculate percentages
conf_matrix_percent1 = conf_matrix1 / conf_matrix1.sum(axis=1, keepdims=True) * 100
conf_matrix_percent1 = np.nan_to_num(conf_matrix_percent1, 0)  # Replace NaNs with 0

# Confusion Matrix TM 2
conf_matrix2 = confusion_matrix(Y_test, tm2.predict(X_test2))
# Calculate percentages
conf_matrix_percent2 = conf_matrix2 / conf_matrix2.sum(axis=1, keepdims=True) * 100
conf_matrix_percent2 = np.nan_to_num(conf_matrix_percent2, 0)  # Replace NaNs with 0

# Plot the confusion matrices with percentages
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Confusion Matrix run 1 (Percentages)
sns.heatmap(conf_matrix_percent1, annot=True, fmt=".1f", cmap="Blues", xticklabels=(0,1,2), yticklabels=(0,1,2), ax=axs[0])
axs[0].set_title("Confusion Matrix run 1 (Percentages)")
axs[0].set_xlabel("Predicted Labels")
axs[0].set_ylabel("True Labels")

# Confusion Matrix run 2 (Percentages)
sns.heatmap(conf_matrix_percent2, annot=True, fmt=".1f", cmap="Blues", xticklabels=(0,1,2), yticklabels=(0,1,2), ax=axs[1])
axs[1].set_title("Confusion Matrix run 2 (Percentages)")
axs[1].set_xlabel("Predicted Labels")
axs[1].set_ylabel("True Labels")

plt.show()

