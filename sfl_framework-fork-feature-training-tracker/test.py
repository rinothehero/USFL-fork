# Creating a Python list for the accuracy values from the table
import numpy as np

# a = [
#     45.87,
#     47.04,
#     45.91,
#     46.11,
#     47.04,
#     45.91,
#     47.17,
#     47.13,
#     45.87,
#     46.04,
#     47.04,
#     45.91,
#     44.42,
#     47.13,
# ]

# standard = 48.22

# diffs_in_percent = [(value - standard) / standard * 100 for value in a]

# print(np.mean(diffs_in_percent))


accuracy_values = [
    [26.05, 33.72, 30.47, 29.13, 35.56],  # ResNet-18, CIFAR-10
    [18.13, 33.60, 21.90, 25.46, 34.23],  # VGG-11, CIFAR-10
    [47.78, 76.25, 69.38, 81.74, 84.79],  # AlexNet, FMNIST
]

diffs_in_percent_across_models = []

for acc in accuracy_values:
    fitfl_acc = acc[4]  # FitFL의 정확도
    # FedAvg (acc[0]) 제외, SOTA 기술들만 비교
    diffs_in_percent = [(fitfl_acc - value) / fitfl_acc * 100 for value in acc[1:4]]
    diffs_in_percent_across_models.append(np.mean(diffs_in_percent))

# 전체 모델 평균 계산
average_diff_in_percent = np.mean(diffs_in_percent_across_models)
print(average_diff_in_percent)
