import matplotlib.pyplot as plt

# heart
clusters = [1, 2, 3, 4, 5, 6, 7]
F1 = [0.8670, 0.9055, 0.9057, 0.9073, 0.9254, 0.9223, 0.9327]
ACC = [0.8533, 0.8967, 0.8919, 0.8973, 0.9194, 0.9140, 0.9251]
AUC = [0.9215, 0.8973, 0.8868, 0.8966, 0.9213, 0.9140, 0.9233]

# cleveland
# clusters = [1, 2, 3, 4, 5, 6, 7]
# F1 = [0.7000, 0.8772, 0.9091, 0.9153, 0.8852, 0.9032, 0.9123]
# ACC = [0.7778, 0.8833, 0.9180, 0.9180, 0.8852, 0.9016, 0.9180]
# AUC = [0.8876, 0.8839, 0.9161, 0.9186, 0.8874, 0.9046, 0.9188]

# Plot
plt.figure(figsize=(6.5, 6))
plt.plot(clusters, F1, marker='o', label='F1')
plt.plot(clusters, ACC, marker='s', label='Accuracy')
plt.plot(clusters, AUC, marker='^', label='AUC')

# Y-axis scaling (not starting from 0)
plt.ylim(0.84, 0.94)
# plt.ylim(0.67, 0.94)
plt.xticks(clusters, fontsize=14)
plt.yticks(fontsize=14)

# Labels and title in English
plt.xlabel('Number of Clusters', fontsize=17)
plt.ylabel('Score', fontsize=17)
# plt.title('Heart Dataset', fontsize=20)
plt.legend(fontsize=15)
plt.grid(True, linestyle='--', alpha=0.4)

plt.show()
