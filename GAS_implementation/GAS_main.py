import sys
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

import copy
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import random
import numpy as np
import datetime
from network import model_selection
from dataset import Dataset, Data_Partition
from utils import (
    calculate_v_value,
    replace_user,
    sample_or_generate_features,
    compute_local_adjustment,
    find_client_with_min_time,
)
from g_measurement import GMeasurementManager, compute_g_score


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experimental parameter settings
iid = False
dirichlet = False
label_dirichlet = True  # Hybrid: shard classes + Dirichlet quantity
shard = 2
alpha = 0.3
min_require_size = 10  # Minimum samples per client for data partitioning
epochs = 300
localEpoch = 5
user_num = 100
user_parti_num = 10
batchSize = 25
lr = 0.001
momentum = 0.0
weight_decay = 0.0
# Training data selection
cifar = True
mnist = False
fmnist = True
cinic = False
cifar100 = False
SVHN = False
# Model selection
use_resnet = True
split_ratio = "quarter"  # Legacy: 'half' or 'quarter' (only if split_layer is None)
split_layer = "layer1.1.bn2"  # Fine-grained: 'layer1', 'layer1.0.bn1', 'layer2', etc.
split_alexnet = "default"
# Random seeds selection
seed_value = 2023
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
clip_grad = True

# Hyperparameter Setting of GAS
Generate = True  # Whether to generate activations
Sample_Frequency = 1  # Sampling frequency
V_Test = True  # Calculate Gradient Dissimilarity
V_Test_Frequency = 1
Accu_Test_Frequency = 1
num_label = 100 if cifar100 else 10

# G Measurement Settings (USFL-style 3-perspective)
G_Measurement = True  # Enable 3-perspective G measurement
G_Measure_Frequency = 10  # Diagnostic round frequency (every N epochs)
G_Measure_Mode = "strict"  # 'strict' (Global Model) or 'realistic' (Individual Models)

# G Measurement Mode: distance-based (default) vs variance-based (SFL-style)
# Usage: python GAS_main.py true  -> variance-based
#        python GAS_main.py false -> distance-based
USE_VARIANCE_G = False
if len(sys.argv) > 1:
    arg = sys.argv[1].strip().lower()
    if arg in {"true", "1", "yes", "y"}:
        USE_VARIANCE_G = True
    elif arg in {"false", "0", "no", "n"}:
        USE_VARIANCE_G = False
    else:
        raise ValueError(
            "First argument must be a boolean (true/false) for USE_VARIANCE_G"
        )

# Simulate real communication environments
WRTT = True  # True for simulation, False for no simulation


"""communication formulation"""


# Initialize client computing capabilities
def generate_computing(user_num):
    # 10**9 ～ 10**10 FLOPs
    return np.random.uniform(10**9, 10**10, user_num)


def generate_position(user_num):
    return np.random.uniform(0.1, 1, user_num)


clients_computing = generate_computing(user_num)
clients_position = generate_position(user_num)

# Calculate the communication rate for each client
w = 10**7
N = 3.981 * 10 ** (-21)
rates = []
for i in range(user_num):
    path_loss = 128.1 + 37.6 * np.log10(clients_position[i])
    h = 10 ** (-path_loss / 10)
    rates.append(w * np.log2(1 + (0.2 * h / (w * N))))

# clients_computing = [3897894735.92771, 9013802066.105328, 6292470299.054264, 2139364841.5386212, 2272071003.1736717,
#                      5211060330.473793, 1198806954.5634105, 7545472413.904353, 5719486080.450004, 5904417149.294411,
#                      5107359343.831957, 5512440381.97626, 4550216975.213522, 2360550729.2126856, 4247876594.1973186,
#                      2458693061.80659, 4041628237.5291243, 2622909526.838994, 4518922610.259291, 1320833893.6367936]
# clients_position = [0.6083754840266261, 0.2831153426857548, 0.38854401189901344, 0.4389074044864103, 0.2656487276448098,
#                     0.19355665271763545, 0.509434502092531, 0.2762774537423048, 0.4406728802978894, 0.9374787633533964,
#                     0.7841437418818485, 0.7936878138929144, 0.6370305015692377, 0.8124590324955729, 0.8293044732903868,
#                     0.9825015031837409, 0.8963067286477621, 0.19882101292753768, 0.8377396846064555, 0.3768516013443233]
# rates = [25948858.23931885, 64992952.13823921, 48181797.74664864, 41864543.70309627, 68413272.50342345,
#          85499773.20790398, 34364550.5456598, 66305169.93391083, 41658828.81171437, 9941368.164306499,
#          15564920.989243941, 15135144.574422104, 23896461.265810642, 14324059.624996226, 13633385.95929576,
#          8730296.888630323, 11205899.934015611, 84048199.78995569, 13300281.995040257, 49783651.93687485]

if WRTT is True:
    print(clients_computing)
    print(clients_position)
    print(rates)


"""Class Definition"""


# Client class definition
class Client:
    def __init__(
        self,
        user_data,
        local_epoch,
        minibatch=0,
        computing=0,
        rate=0,
        time=0,
        weight_count=1,
    ):
        self.user_data = user_data
        self.dataloader_iter = iter(user_data)
        self.local_epoch = local_epoch
        self.count = 0
        # Calculation of time
        self.minibatch = minibatch
        self.computing = computing
        self.rate = rate
        self.time = time
        # weight
        self.weight_count = weight_count

    def increment_counter(self):
        # record the number of local iterations
        self.count += 1
        if self.count == self.local_epoch:
            self.count = 0
            return True
        return False

    def train_one_iteration(self):
        try:
            data = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.user_data)
            data = next(self.dataloader_iter)
        return data

    # Calculation of time
    def model_process(self):
        # AlexNet
        workload = 5603328  # Workload of one image FLOPs
        # workload *= self.minibatch  # parallel computations
        self.time += workload / self.computing

    def transmit_activation(self):
        # AlexNet
        activation = 131072  # Size of an activation value (bit) (64 * 8 * 8 * 32)
        activation *= self.minibatch
        self.time += activation / self.rate

    def transmit_model(self):
        # AlexNet
        model_volume = 620544
        self.time += model_volume / self.rate


# IncrementalStats class for maintaining mean and variance
class IncrementalStats:
    def __init__(self, device, diagonal=False):
        self.device = device
        self.diagonal = (
            diagonal  # True for ResNet (memory efficient), False for AlexNet
        )
        self.means = {}
        self.variances = {}
        self.weight = {}
        self.counts = {}

    def update(self, new_mean, new_var_or_cov, new_weight, label):
        """
        Update the weighted mean and variance/covariance of the features for the given label.
        :param new_mean: Mean vector of type torch.Tensor
        :param new_var_or_cov: Variance vector (diagonal) or Covariance matrix depending on self.diagonal
        :param new_weight: Weight for this update
        :param label: Label
        """
        regularization_term = 1e-5

        if label not in self.means:
            self.means[label] = new_mean.to(self.device)
            self.variances[label] = new_var_or_cov.to(self.device)
            self.counts[label] = 1
            self.weight[label] = new_weight
        else:
            old_mean = self.means[label]
            old_var = self.variances[label]
            old_weight = self.weight[label]
            self.weight[label] = old_weight + new_weight
            decay_factor = old_weight / self.weight[label]

            # Update mean
            self.means[label] = decay_factor * old_mean + (1 - decay_factor) * new_mean

            if self.diagonal:
                # Diagonal variance (for ResNet - memory efficient)
                self.variances[label] = (
                    decay_factor * (old_var + (self.means[label] - old_mean) ** 2)
                    + (1 - decay_factor)
                    * (new_var_or_cov + (self.means[label] - new_mean) ** 2)
                    + regularization_term
                )
            else:
                # Full covariance matrix (for AlexNet - more accurate)
                n = new_mean.shape[0]
                I = torch.eye(n).to(self.device)
                self.variances[label] = (
                    decay_factor
                    * (
                        old_var
                        + torch.outer(
                            self.means[label] - old_mean, self.means[label] - old_mean
                        )
                    )
                    + (1 - decay_factor)
                    * (
                        new_var_or_cov
                        + torch.outer(
                            self.means[label] - new_mean, self.means[label] - new_mean
                        )
                    )
                    + regularization_term * I
                )
            self.counts[label] += 1

    def get_stats(self, label):
        """
        Get the weighted mean and variance for the given label.
        :param label: Label
        :return: Weighted mean and variance of type torch.Tensor
        """
        return self.means.get(label, None), self.variances.get(label, None)


"""Main Train"""

# Data loading and preprocessing
alldata, alllabel, test_set, transform = Dataset(
    cifar=cifar, mnist=mnist, fmnist=fmnist, cinic=cinic, cifar100=cifar100, SVHN=SVHN
)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=128, shuffle=True)
train_index = np.arange(0, len(alldata))
random.shuffle(train_index)
train_img = np.array(alldata)[train_index]
train_label = np.array(alllabel)[train_index]
users_data = Data_Partition(
    iid,
    dirichlet,
    train_img,
    train_label,
    transform,
    user_num,
    batchSize,
    alpha,
    shard,
    drop=False,
    classOfLabel=num_label,
    label_dirichlet=label_dirichlet,
    min_require_size=min_require_size,
)

# Model initialization
user_model, server_model = model_selection(
    cifar,
    mnist,
    fmnist,
    cinic=cinic,
    split=True,
    cifar100=cifar100,
    SVHN=SVHN,
    resnet=use_resnet,
    split_ratio=split_ratio,
    split_layer=split_layer,
    split_alexnet=split_alexnet,
)
user_model.to(device)
server_model.to(device)
userParam = copy.deepcopy(user_model.state_dict())
optimizer_down = torch.optim.SGD(
    user_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
optimizer_up = torch.optim.SGD(
    server_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()

# Initialize clients
if WRTT is True:
    clients = [
        Client(users_data[i], localEpoch, batchSize, clients_computing[i], rates[i], 0)
        for i in range(user_num)
    ]
else:
    clients = [Client(users_data[i], localEpoch) for i in range(user_num)]
stats = IncrementalStats(
    device=device, diagonal=use_resnet
)  # ResNet uses diagonal, AlexNet uses full covariance
condensed_data = {c: None for c in range(num_label)}
train_begin_time = datetime.datetime.now()

# Initialize G Measurement Manager (USFL-style 3-perspective)
g_manager = None
full_train_loader = None
if G_Measurement:
    g_manager = GMeasurementManager(device, measure_frequency=G_Measure_Frequency)
    # Create a DataLoader for full training data (for Oracle computation with proper transforms)
    from dataset import TrainSet

    full_train_set = TrainSet(train_img, train_label, transform)
    full_train_loader = dataloader.DataLoader(
        dataset=full_train_set, batch_size=batchSize, shuffle=False, drop_last=False
    )
    print(f"[G Measurement] Initialized. Frequency: every {G_Measure_Frequency} epochs")

    g_measure_state = {
        "active": False,
        "epoch": None,
        "client_order": [],
        "client_grads": {},
        "client_split_grads": {},
        "client_batch_sizes": {},
        "server_grads": [],
        "server_batch_sizes": [],
    }
else:
    g_measure_state = None


def finalize_g_measurement(g_measure_state, g_manager, user_parti_num):
    if g_measure_state is None or not g_measure_state["active"]:
        return False
    if g_manager is None or g_manager.oracle_grads is None:
        return False

    if (
        len(g_measure_state["client_grads"]) < user_parti_num
        or len(g_measure_state["server_grads"]) < user_parti_num
    ):
        return False

    def flatten_grad_list(grad_list):
        return torch.cat([g.flatten().float() for g in grad_list])

    per_client_g = {}
    per_client_vecs = {}
    for client_id in g_measure_state["client_order"]:
        current_client_grads = g_measure_state["client_grads"].get(client_id)
        if current_client_grads is None:
            continue
        g_details = compute_g_score(
            g_manager.oracle_grads["client"],
            current_client_grads,
            return_details=True,
        )
        per_client_g[client_id] = g_details
        per_client_vecs[client_id] = flatten_grad_list(current_client_grads)

    if per_client_g:
        avg_client_g = sum(gd["G"] for gd in per_client_g.values()) / len(per_client_g)
        avg_g_rel = sum(gd["G_rel"] for gd in per_client_g.values()) / len(per_client_g)
    else:
        avg_client_g = float("nan")
        avg_g_rel = float("nan")

    server_g_list = []
    server_vecs = []
    for server_grad in g_measure_state["server_grads"]:
        server_g_list.append(
            compute_g_score(g_manager.oracle_grads["server"], server_grad)
        )
        server_vecs.append(flatten_grad_list(server_grad))
    if server_g_list:
        avg_server_g = sum(server_g_list) / len(server_g_list)
    else:
        avg_server_g = float("nan")

    variance_client_g = float("nan")
    variance_client_g_rel = float("nan")
    if USE_VARIANCE_G and per_client_vecs:
        oracle_client_vec = flatten_grad_list(g_manager.oracle_grads["client"])
        total_weight = sum(g_measure_state["client_batch_sizes"].values())
        if total_weight > 0:
            Vc = 0.0
            denom_c = 0.0
            for client_id, vec in per_client_vecs.items():
                weight = (
                    g_measure_state["client_batch_sizes"].get(client_id, 0)
                    / total_weight
                )
                diff = vec - oracle_client_vec
                Vc += weight * torch.dot(diff, diff).item()
                denom_c += weight * torch.dot(vec, vec).item()
            variance_client_g = Vc**0.5
            variance_client_g_rel = (
                (Vc / denom_c) ** 0.5 if denom_c > 0 else float("nan")
            )

    variance_server_g = float("nan")
    variance_server_g_rel = float("nan")
    if USE_VARIANCE_G and server_vecs:
        oracle_server_vec = flatten_grad_list(g_manager.oracle_grads["server"])
        total_weight = sum(g_measure_state["server_batch_sizes"])
        if total_weight > 0:
            Vs = 0.0
            denom_s = 0.0
            for vec, batch_size in zip(
                server_vecs, g_measure_state["server_batch_sizes"]
            ):
                weight = batch_size / total_weight
                diff = vec - oracle_server_vec
                Vs += weight * torch.dot(diff, diff).item()
                denom_s += weight * torch.dot(vec, vec).item()
            variance_server_g = Vs**0.5
            variance_server_g_rel = (
                (Vs / denom_s) ** 0.5 if denom_s > 0 else float("nan")
            )

    split_g = float("nan")
    if (
        g_measure_state["client_split_grads"]
        and g_manager.oracle_grads.get("split") is not None
    ):
        split_stack = torch.stack(list(g_measure_state["client_split_grads"].values()))
        split_avg = split_stack.mean(dim=0)
        split_g = compute_g_score(g_manager.oracle_grads["split"], split_avg)

    print(f"[G Measurement] Epoch {g_measure_state['epoch']} - Per-Client G:")
    for cid, gd in per_client_g.items():
        print(
            f"  Client {cid}: ||oracle||={gd['oracle_norm']:.4f}, ||current||={gd['current_norm']:.4f}, "
            f"G={gd['G']:.4f}, G_rel={gd['G_rel']:.1f}%"
        )
    print(f"  Average: G={avg_client_g:.4f}, G_rel={avg_g_rel:.1f}%")

    for idx, g_val in enumerate(server_g_list):
        print(f"[G Measurement] Server update {idx}: G={g_val:.6f}")
    print(f"[G Measurement] Server Average G = {avg_server_g:.6f}")

    if USE_VARIANCE_G:
        print(
            f"[G Measurement] Variance Client G = {variance_client_g:.6f}, "
            f"G_rel = {variance_client_g_rel:.6f}"
        )
        print(
            f"[G Measurement] Variance Server G = {variance_server_g:.6f}, "
            f"G_rel = {variance_server_g_rel:.6f}"
        )

    if USE_VARIANCE_G:
        g_manager.g_history["client_g"].append(variance_client_g)
        g_manager.g_history["server_g"].append(variance_server_g)
    else:
        g_manager.g_history["client_g"].append(avg_client_g)
        g_manager.g_history["server_g"].append(avg_server_g)
    g_manager.g_history["split_g"].append(split_g)

    g_manager.oracle_grads = None
    g_measure_state["active"] = False
    g_measure_state["epoch"] = None
    g_measure_state["client_order"] = []
    g_measure_state["client_grads"].clear()
    g_measure_state["client_split_grads"].clear()
    g_measure_state["client_batch_sizes"].clear()
    g_measure_state["server_grads"].clear()
    g_measure_state["server_batch_sizes"].clear()

    return True


# Training loop
total_accuracy = []
total_v_value = []
local_models_time = []
time_record = []
epoch = 0
order = np.random.choice(
    range(user_num), user_parti_num, replace=False
)  # 初始选择的用户
if WRTT is True:  # initialize training time
    for i in order:
        clients[i].model_process()
        clients[i].transmit_activation()
usersParam = [copy.deepcopy(userParam) for _ in range(user_parti_num)]
concat_features = None
concat_labels = None
concat_weight_counts = None
sumClientParam = None
feature_shape = None
count_concat = 0  # activation cache
count_local = 0  # local-side model cache
local_epoch = 0  # total number of local iterations
total_weight_count = 1

# generate local logit adjustment for each client
logit_local_adjustments = []
for i in range(user_num):
    logit_local_adjustments.append(compute_local_adjustment(users_data[i], device))

while epoch != epochs:
    user_model.train()
    server_model.train()
    # select a client
    if WRTT is True:
        selected_client = find_client_with_min_time(clients, order)
    else:
        selected_client = np.random.choice(order)
    user_model.load_state_dict(
        usersParam[np.where(order == selected_client)[0][0]], strict=True
    )
    # train
    images, labels = clients[selected_client].train_one_iteration()
    images = images.to(device)
    labels = labels.to(device)
    split_layer_output = user_model(images)

    # Handle tuple output from fine-grained split (activation, identity)
    if isinstance(split_layer_output, tuple):
        activation, identity = split_layer_output
        features_for_stats = activation  # Use activation for statistics
    else:
        activation = split_layer_output
        identity = None
        features_for_stats = activation

    if g_measure_state is not None and g_measure_state["active"]:
        activation.retain_grad()

    if feature_shape is None:
        feature_shape = features_for_stats[0].shape

    # Define the weight vector to record the weight of each activation value
    weight_count = clients[selected_client].weight_count
    weight_vector = torch.tensor([weight_count] * features_for_stats.size(0))

    # generate concatenated activation
    features = features_for_stats.detach()
    count_concat += 1
    if concat_features is None:
        concat_features = features
        concat_labels = labels
        concat_weight_counts = weight_vector
    else:
        concat_features = torch.cat(
            (concat_features, features_for_stats.detach()), dim=0
        )
        concat_labels = torch.cat((concat_labels, labels), dim=0)
        concat_weight_counts = torch.cat((concat_weight_counts, weight_vector), dim=0)

    # Update weight of clients
    clients[selected_client].weight_count = clients[selected_client].weight_count + 1

    # client-side model update
    local_output = server_model(
        split_layer_output
    )  # Pass original output (tuple or tensor)

    # localLoss = criterion(local_output, labels)
    localLoss = criterion(
        local_output + logit_local_adjustments[selected_client], labels.long()
    )
    optimizer_down.zero_grad()
    localLoss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(parameters=user_model.parameters(), max_norm=10)

    if g_measure_state is not None and g_measure_state["active"]:
        if (
            selected_client not in g_measure_state["client_grads"]
            and len(g_measure_state["client_grads"]) < user_parti_num
        ):
            g_measure_state["client_order"].append(selected_client)
            g_measure_state["client_grads"][selected_client] = [
                p.grad.clone().cpu()
                if p.grad is not None
                else torch.zeros_like(p).cpu()
                for p in user_model.parameters()
            ]
            g_measure_state["client_batch_sizes"][selected_client] = labels.size(0)
            if activation.grad is not None:
                g_measure_state["client_split_grads"][selected_client] = (
                    activation.grad.mean(dim=0).clone().cpu()
                )
        finalize_g_measurement(g_measure_state, g_manager, user_parti_num)

    optimizer_down.step()
    usersParam[np.where(order == selected_client)[0][0]] = copy.deepcopy(
        user_model.state_dict()
    )
    if WRTT is True:  # Record the time of the backward pass
        clients[selected_client].model_process()

    """Activations generation and server-side model update"""
    if count_concat == user_parti_num:
        # print("local_epoch: " + str(local_epoch))
        local_epoch += 1
        # update activation distributions
        unique_labels, counts = concat_labels.unique(
            return_counts=True
        )  # Count how many of each label

        label_weights = {}
        concat_weight_counts = concat_weight_counts.to(device)
        for label in unique_labels:
            mask = concat_labels == label
            weights_of_label = concat_weight_counts[mask].float()
            label_weights[label.item()] = weights_of_label.sum().item()

        # Calculate mean and variance/covariance
        flatten_features = concat_features.flatten(start_dim=1)
        for label in unique_labels:
            mask = concat_labels == label
            features_of_label = flatten_features[mask]
            weights_of_label = concat_weight_counts[mask].float()
            total_weight = weights_of_label.sum()
            mean_feature = (
                torch.sum(features_of_label * weights_of_label[:, None], dim=0)
                / total_weight
            )
            centered_features = features_of_label - mean_feature

            if use_resnet:
                # Diagonal variance for ResNet (memory efficient)
                var_vector = (
                    torch.sum((centered_features**2) * weights_of_label[:, None], dim=0)
                    / total_weight
                )
                stats.update(
                    mean_feature, var_vector, label_weights[label.item()], label.item()
                )
            else:
                # Full covariance matrix for AlexNet (more accurate)
                cov_matrix = (
                    torch.matmul(
                        (centered_features * weights_of_label[:, None]).T,
                        centered_features,
                    )
                    / total_weight
                )
                stats.update(
                    mean_feature, cov_matrix, label_weights[label.item()], label.item()
                )
        if Generate is True:
            # Activations generation
            if local_epoch % Sample_Frequency == 0:
                # Ensure that all labels have mean and variance
                all_labels_have_stats = True
                for label in range(num_label):
                    if stats.get_stats(label) == (None, None):
                        all_labels_have_stats = False
                        break
                if all_labels_have_stats:
                    concat_features, concat_labels = sample_or_generate_features(
                        concat_features,
                        concat_labels,
                        batchSize,
                        num_label,
                        feature_shape,
                        device,
                        stats,
                        diagonal=use_resnet,
                    )

        # server-side model update
        for param in server_model.parameters():
            param.requires_grad = True
        final_output = server_model(concat_features)
        loss = criterion(final_output, concat_labels.long())
        optimizer_up.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(
                parameters=server_model.parameters(), max_norm=10
            )

        if g_measure_state is not None and g_measure_state["active"]:
            if len(g_measure_state["server_grads"]) < user_parti_num:
                g_measure_state["server_grads"].append(
                    [
                        p.grad.clone().cpu()
                        if p.grad is not None
                        else torch.zeros_like(p).cpu()
                        for p in server_model.parameters()
                    ]
                )
                g_measure_state["server_batch_sizes"].append(concat_labels.size(0))
            finalize_g_measurement(g_measure_state, g_manager, user_parti_num)

        optimizer_up.step()
        if V_Test is True:
            concat_labels_V = copy.deepcopy(concat_labels)
            concat_features_V = copy.deepcopy(concat_features)
        count_concat = 0
        concat_labels = None
        concat_features = None
        concat_weight_counts = None

    # client-side models aggregation
    replace = clients[selected_client].increment_counter()
    if replace:  # If local iterations are completed, select a new client
        count_local += 1
        if WRTT is True:  # Record the time of model upload
            clients[selected_client].transmit_model()
            local_models_time.append(clients[selected_client].time)
        if sumClientParam is None:
            sumClientParam = usersParam[np.where(order == selected_client)[0][0]]
            for key in usersParam[np.where(order == selected_client)[0][0]]:
                sumClientParam[key] = usersParam[
                    np.where(order == selected_client)[0][0]
                ][key] * (1 / user_parti_num)
        else:
            for key in usersParam[np.where(order == selected_client)[0][0]]:
                sumClientParam[key] += usersParam[
                    np.where(order == selected_client)[0][0]
                ][key] * (1 / user_parti_num)
        if (
            count_local == user_parti_num
        ):  # Update the client model if the buffer is full
            total_weight_count += local_epoch
            userParam = copy.deepcopy(sumClientParam)
            sumClientParam = None
            count_local = 0

            test_flag = (epoch + 1) % Accu_Test_Frequency == 0
            epoch += 1

            if WRTT:
                if test_flag:
                    time_record.append(max(local_models_time))
                    print("Time: " + str(time_record[-1]))
                local_models_time = []

            # Accuracy test
            if test_flag:
                user_model.eval()
                server_model.eval()
                user_model.load_state_dict(userParam, strict=True)
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = user_model(images)
                        output = server_model(output)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                total_accuracy.append(accuracy)
                print("Global iteration: " + str(epoch))
                print("Accuracy: " + str(total_accuracy[-1]))
                print()

            # V test
            if V_Test and (epoch + 1) % V_Test_Frequency == 0:
                v_value = calculate_v_value(
                    server_model,
                    user_model,
                    concat_features_V,
                    concat_labels_V,
                    test_loader,
                    criterion,
                    device,
                )
                print(f"Epoch {epoch + 1}, V Value: {v_value}")
                total_v_value.append(v_value)

            # G Measurement (Async-faithful 3-perspective)
            if (
                G_Measurement
                and g_manager is not None
                and g_manager.should_measure(epoch)
                and g_measure_state is not None
            ):
                from g_measurement import assert_param_name_alignment

                user_model.load_state_dict(userParam, strict=True)
                g_manager.compute_oracle(
                    user_model, server_model, full_train_loader, criterion
                )
                if g_manager.oracle_grads is not None:
                    assert_param_name_alignment(
                        g_manager.oracle_grads["client_names"],
                        user_model,
                        "client",
                    )
                    assert_param_name_alignment(
                        g_manager.oracle_grads["server_names"],
                        server_model,
                        "server",
                    )

                g_measure_state["active"] = True
                g_measure_state["epoch"] = epoch
                g_measure_state["client_order"] = []
                g_measure_state["client_grads"].clear()
                g_measure_state["client_split_grads"].clear()
                g_measure_state["server_grads"].clear()

                print(
                    f"[G Measurement] Epoch {epoch}: capturing async gradients (client/server)"
                )

        # select new client
        index = np.where(order == selected_client)[0][0]
        usersParam[index] = userParam

        if WRTT is True:  # Initialize the time for the new client
            begin_time = clients[selected_client].time
            order = replace_user(order, selected_client, user_num)
            clients[order[index]].weight_count = total_weight_count
            clients[order[index]].time = begin_time
            clients[order[index]].model_process()
            clients[order[index]].transmit_activation()
        else:
            order = replace_user(order, selected_client, user_num)
            clients[order[index]].weight_count = total_weight_count
    else:
        if (
            WRTT is True
        ):  # Record the training time if the client continues with local iterations
            clients[selected_client].model_process()
            clients[selected_client].transmit_activation()

# Output results
print(time_record)
print(total_accuracy)
print(total_v_value)
time_record_str = ", ".join(str(x) for x in time_record)
total_accuracy_str = ", ".join(str(x) for x in total_accuracy)
total_v_value_str = ", ".join(str(x) for x in total_v_value)
print("time = [" + time_record_str + "]")
print("GAS = [" + total_accuracy_str + "]")

end_time = datetime.datetime.now()
begin_time_str = train_begin_time.strftime("%Y-%m-%d %H:%M:%S")
end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

selectDataset = (
    "cifar10"
    if cifar
    else "mnist"
    if mnist
    else "fmnist"
    if fmnist
    else "cinic"
    if cinic
    else "cifar100"
    if cifar100
    else "SVHN"
    if SVHN
    else "None"
)
selectMethod = "Generative Activation-Aided" if Generate else "Original"
IfCilp = "clip" if clip_grad else "not clip"

# Generate filename with timestamp
timestamp_str = end_time.strftime("%Y%m%d_%H%M%S")
output_filename = f"GAS_main_{timestamp_str}.txt"

with open(output_filename, "w") as f:
    # ===== Configuration Summary =====
    f.write("=" * 60 + "\n")
    f.write("EXPERIMENT CONFIGURATION\n")
    f.write("=" * 60 + "\n")

    # Model settings
    model_type = "ResNet-18" if use_resnet else "AlexNet"
    f.write(f"Model: {model_type}\n")
    if use_resnet:
        f.write(f"  split_layer: {split_layer}\n")
        f.write(f"  split_ratio: {split_ratio}\n")
    else:
        f.write(f"  split_alexnet: {split_alexnet}\n")

    # Data settings
    f.write(f"\nData Settings:\n")
    f.write(f"  Dataset: {selectDataset}\n")
    f.write(
        f"  IID: {iid}, Dirichlet: {dirichlet}, Label_Dirichlet: {label_dirichlet}\n"
    )
    f.write(f"  Alpha: {alpha}, Shard: {shard}, Min_require_size: {min_require_size}\n")

    # Training settings
    f.write(f"\nTraining Settings:\n")
    f.write(f"  Epochs: {epochs}, LocalEpoch: {localEpoch}\n")
    f.write(f"  Users: {user_num}, Participating: {user_parti_num}\n")
    f.write(f"  BatchSize: {batchSize}, LR: {lr}\n")
    f.write(f"  Momentum: {momentum}, Weight_decay: {weight_decay}\n")
    f.write(f"  Clip_grad: {clip_grad}\n")

    # GAS settings
    f.write(f"\nGAS Settings:\n")
    f.write(f"  Generate: {Generate}, Sample_Frequency: {Sample_Frequency}\n")
    f.write(f"  V_Test: {V_Test}, V_Test_Frequency: {V_Test_Frequency}\n")
    f.write(
        f"  G_Measurement: {G_Measurement}, G_Measure_Frequency: {G_Measure_Frequency}\n"
    )
    f.write(f"  WRTT: {WRTT}\n")

    # Other
    f.write(f"\nOther:\n")
    f.write(f"  Seed: {seed_value}\n")
    f.write(f"  Method: {selectMethod}\n")
    f.write("=" * 60 + "\n\n")

    # ===== Results =====
    f.write("RESULTS\n")
    f.write("-" * 60 + "\n")

    if V_Test is True:
        f.write(f"Test Frequency is {V_Test_Frequency}; \n")
        f.write("Gradient Dissimilarity = [" + total_v_value_str + "]\n")

    if WRTT is True:
        clients_computing_str = ", ".join(str(x) for x in clients_computing)
        clients_position_str = ", ".join(str(x) for x in clients_position)
        rates_str = ", ".join(str(x) for x in rates)
        f.write("clients computing = [" + clients_computing_str + "]\n")
        f.write("clients position = [" + clients_position_str + "]\n")
        f.write("clients rates = [" + rates_str + "]\n")
        f.write("time = [" + time_record_str + "]\n")

    if G_Measurement and g_manager is not None:
        client_g_str = ", ".join(f"{x:.6f}" for x in g_manager.g_history["client_g"])
        server_g_str = ", ".join(f"{x:.6f}" for x in g_manager.g_history["server_g"])
        split_g_str = ", ".join(f"{x:.6f}" for x in g_manager.g_history["split_g"])
        f.write(f"Client G = [{client_g_str}]\n")
        f.write(f"Server G = [{server_g_str}]\n")
        f.write(f"Split G = [{split_g_str}]\n")

    f.write(f"{begin_time_str} ~ {end_time_str};\n")
    f.write("GAS = [" + total_accuracy_str + "]\n")
