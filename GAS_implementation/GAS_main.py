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
import json
from typing import Dict
from network import model_selection, load_torchvision_resnet18_init
from dataset import Dataset, Data_Partition
from utils import (
    calculate_v_value,
    replace_user,
    sample_or_generate_features,
    compute_local_adjustment,
    find_client_with_min_time,
)
from g_measurement import GMeasurementManager, compute_g_score
from drift_measurement import DriftMeasurementTracker


def _set_batchnorm_eval(module: nn.Module) -> Dict[str, bool]:
    states: Dict[str, bool] = {}
    for name, child in module.named_modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            states[name] = child.training
            child.eval()
    return states


def _restore_batchnorm(module: nn.Module, states: Dict[str, bool]) -> None:
    for name, child in module.named_modules():
        if name in states:
            child.train(states[name])


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
batchSize = 50
lr = 0.001
momentum = 0.0
weight_decay = 0.0
# Training data selection
cifar = True
mnist = False
fmnist = False
cinic = False
cifar100 = False
SVHN = False
# Model selection
use_resnet = True
use_resnet_image_style = True  # True to match torchvision ResNet18 stem
split_ratio = "quarter"  # Legacy: 'half' or 'quarter' (only if split_layer is None)
split_layer = "layer1.1.bn2"  # Fine-grained: 'layer1', 'layer1.0.bn1', 'layer2', etc.
split_alexnet = "default"

# Optional env overrides for experiment runner


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


dataset_override = _env_str("GAS_DATASET", "")
if dataset_override:
    cifar = dataset_override == "cifar10"
    mnist = dataset_override == "mnist"
    fmnist = dataset_override == "fmnist"
    cinic = dataset_override == "cinic"
    cifar100 = dataset_override == "cifar100"
    SVHN = dataset_override == "svhn"

model_override = _env_str("GAS_MODEL", "")
if model_override:
    if model_override == "resnet18":
        use_resnet = True
        use_resnet_image_style = _env_bool(
            "GAS_USE_RESNET_IMAGE_STYLE", use_resnet_image_style
        )
    elif model_override == "alexnet":
        use_resnet = False
        use_resnet_image_style = False

batchSize = _env_int("GAS_BATCH_SIZE", batchSize)
shard = _env_int("GAS_LABELS_PER_CLIENT", shard)
alpha = _env_float("GAS_DIRICHLET_ALPHA", alpha)
min_require_size = _env_int("GAS_MIN_REQUIRE_SIZE", min_require_size)
epochs = _env_int("GAS_GLOBAL_EPOCHS", epochs)
localEpoch = _env_int("GAS_LOCAL_EPOCHS", localEpoch)
user_num = _env_int("GAS_TOTAL_CLIENTS", user_num)
user_parti_num = _env_int("GAS_CLIENTS_PER_ROUND", user_parti_num)
lr = _env_float("GAS_LR", lr)
momentum = _env_float("GAS_MOMENTUM", momentum)
split_layer = _env_str("GAS_SPLIT_LAYER", split_layer)

# Random seeds selection
seed_value = _env_int("GAS_SEED", 2023)
torch.manual_seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
clip_grad = True

# Feature: Use Full Epochs (like sfl-framework)
use_full_epochs = _env_bool("GAS_USE_FULL_EPOCHS", False)
if "--use-full-epochs" in sys.argv:
    use_full_epochs = True

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
G_Measurement_Accumulation = "single"  # "single" (1-step) | "k_batch" (first K) | "accumulated" (full round)
G_Measurement_K = 5  # Number of batches for k_batch mode

# Drift Measurement Settings (SCAFFOLD-style client drift tracking)
DRIFT_MEASUREMENT = _env_bool("GAS_DRIFT_MEASUREMENT", False)
DRIFT_SAMPLE_INTERVAL = _env_int("GAS_DRIFT_SAMPLE_INTERVAL", 1)  # 1 = every step

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

USE_SFL_TRANSFORM = "--sfl-transform" in sys.argv
USE_SFL_ORACLE = True
if "--legacy-oracle" in sys.argv:
    USE_SFL_ORACLE = False

USE_TORCHVISION_INIT = "--torchvision-init" in sys.argv

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
    cifar=cifar,
    mnist=mnist,
    fmnist=fmnist,
    cinic=cinic,
    cifar100=cifar100,
    SVHN=SVHN,
    use_sfl_transform=USE_SFL_TRANSFORM,
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
    resnet_image_style=use_resnet_image_style,
    split_ratio=split_ratio,
    split_layer=split_layer,
    split_alexnet=split_alexnet,
)
if USE_TORCHVISION_INIT and use_resnet_image_style:
    user_model, server_model = load_torchvision_resnet18_init(
        user_model, server_model, split_layer=split_layer or "layer2", image_style=True
    )
user_model.to(device)
server_model.to(device)
full_model = None
if USE_SFL_ORACLE:
    full_model = model_selection(
        cifar,
        mnist,
        fmnist,
        cinic=cinic,
        split=False,
        cifar100=cifar100,
        SVHN=SVHN,
        resnet=use_resnet,
        resnet_image_style=use_resnet_image_style,
    )
    if full_model is not None:
        full_model.to(device)
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
    clients = []
    for i in range(user_num):
        # Determine client steps: full epochs or fixed local iteration count
        if use_full_epochs:
            # users_data[i] is a DataLoader, len() gives number of batches
            client_steps = len(users_data[i]) * localEpoch
            if i == 0:  # Print log only for the first client
                print(
                    f"[GAS][Client {i}] Configured for full epochs: {localEpoch} epochs * {len(users_data[i])} batches = {client_steps} iterations"
                )
        else:
            client_steps = localEpoch

        clients.append(
            Client(
                users_data[i],
                client_steps,
                batchSize,
                clients_computing[i],
                rates[i],
                0,
            )
        )
else:
    clients = []
    for i in range(user_num):
        if use_full_epochs:
            client_steps = len(users_data[i]) * localEpoch
            if i == 0:
                print(
                    f"[GAS][Client {i}] Configured for full epochs: {localEpoch} epochs * {len(users_data[i])} batches = {client_steps} iterations"
                )
        else:
            client_steps = localEpoch

        clients.append(Client(users_data[i], client_steps))

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
        dataset=full_train_set,
        batch_size=batchSize * user_parti_num,
        shuffle=False,
        drop_last=False,
    )
    print(f"[G Measurement] Initialized. Frequency: every {G_Measure_Frequency} epochs")
    print(f"[G Measurement] Accumulation mode: {G_Measurement_Accumulation}")

    g_measure_state = {
        "active": False,
        "epoch": None,
        "client_order": [],
        "client_grads": {},
        "client_split_grads": {},
        "client_batch_sizes": {},
        "server_grads": [],
        "server_batch_sizes": [],
        # Accumulated/K-batch mode: gradient sum and sample count
        "accumulated_client_grads": {},  # {client_id: [grad_sum_tensors]}
        "accumulated_client_samples": {},  # {client_id: total_samples}
        "accumulated_client_batch_counts": {},  # {client_id: batch_count} for k_batch
        "accumulated_server_grads": None,  # [grad_sum_tensors]
        "accumulated_server_samples": 0,
        "accumulated_server_batch_count": 0,  # for k_batch mode
    }
else:
    g_measure_state = None

# Initialize Drift Measurement Tracker
drift_tracker = None
if DRIFT_MEASUREMENT:
    drift_tracker = DriftMeasurementTracker(
        sample_interval=DRIFT_SAMPLE_INTERVAL,
        device=str(device),
    )
    print(f"[Drift Measurement] Initialized. Sample interval: {DRIFT_SAMPLE_INTERVAL}")


def finalize_g_measurement(g_measure_state, g_manager, user_parti_num):
    if g_measure_state is None or not g_measure_state["active"]:
        return False
    if g_manager is None or g_manager.oracle_grads is None:
        return False

    # Check readiness based on accumulation mode
    if G_Measurement_Accumulation in ("accumulated", "k_batch"):
        # Accumulated/K-batch mode: check if we have accumulated data for all clients and server
        if (
            len(g_measure_state["accumulated_client_grads"]) < user_parti_num
            or g_measure_state["accumulated_server_grads"] is None
        ):
            return False
    else:
        # Single mode: original check
        if (
            len(g_measure_state["client_grads"]) < user_parti_num
            or len(g_measure_state["server_grads"]) < 1
        ):
            return False

    def flatten_grad_list(grad_list):
        return torch.cat([g.flatten().float() for g in grad_list])

    per_client_g = {}
    per_client_vecs = {}

    if G_Measurement_Accumulation in ("accumulated", "k_batch"):
        # Accumulated/K-batch mode: compute average gradients and use them
        for client_id in g_measure_state["client_order"]:
            accumulated_grads = g_measure_state["accumulated_client_grads"].get(client_id)
            total_samples = g_measure_state["accumulated_client_samples"].get(client_id, 1)
            if accumulated_grads is None:
                continue
            # Compute average gradient (divide by total samples)
            avg_client_grads = [g / total_samples for g in accumulated_grads]
            g_details = compute_g_score(
                g_manager.oracle_grads["client"],
                avg_client_grads,
                return_details=True,
            )
            per_client_g[client_id] = g_details
            per_client_vecs[client_id] = flatten_grad_list(avg_client_grads)
            # Store for batch_sizes (use total samples)
            g_measure_state["client_batch_sizes"][client_id] = total_samples
        mode_str = "K-batch" if G_Measurement_Accumulation == "k_batch" else "Accumulated"
        print(f"[G Measurement] {mode_str} mode: averaged client gradients from {len(per_client_g)} clients")
    else:
        # Single mode: original logic
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
        avg_d_cosine = sum(gd["D_cosine"] for gd in per_client_g.values()) / len(
            per_client_g
        )
    else:
        avg_client_g = float("nan")
        avg_g_rel = float("nan")
        avg_d_cosine = float("nan")

    server_g_list = []
    server_vecs = []

    if G_Measurement_Accumulation in ("accumulated", "k_batch"):
        # Accumulated/K-batch mode: compute average server gradient
        accumulated_server = g_measure_state["accumulated_server_grads"]
        total_server_samples = g_measure_state["accumulated_server_samples"]
        if accumulated_server is not None and total_server_samples > 0:
            avg_server_grads = [g / total_server_samples for g in accumulated_server]
            g_details = compute_g_score(
                g_manager.oracle_grads["server"],
                avg_server_grads,
                return_details=True,
            )
            server_g_list.append(g_details)
            server_vecs.append(flatten_grad_list(avg_server_grads))
            g_measure_state["server_batch_sizes"] = [total_server_samples]
        print(f"[G Measurement] Accumulated mode: averaged server gradients from {total_server_samples} samples")
    else:
        # Single mode: original logic
        for server_grad in g_measure_state["server_grads"]:
            g_details = compute_g_score(
                g_manager.oracle_grads["server"],
                server_grad,
                return_details=True,
            )
            server_g_list.append(g_details)
            server_vecs.append(flatten_grad_list(server_grad))
    if server_g_list:
        avg_server_g = sum(gd["G"] for gd in server_g_list) / len(server_g_list)
        avg_server_g_rel = sum(gd["G_rel"] for gd in server_g_list) / len(server_g_list)
        avg_server_d = sum(gd["D_cosine"] for gd in server_g_list) / len(server_g_list)
    else:
        avg_server_g = float("nan")
        avg_server_g_rel = float("nan")
        avg_server_d = float("nan")

    variance_client_g = float("nan")
    variance_client_g_rel = float("nan")
    if USE_VARIANCE_G and per_client_vecs:
        oracle_client_vec = flatten_grad_list(g_manager.oracle_grads["client"])
        total_weight = sum(g_measure_state["client_batch_sizes"].values())
        if total_weight > 0:
            Vc = 0.0
            for client_id, vec in per_client_vecs.items():
                weight = (
                    g_measure_state["client_batch_sizes"].get(client_id, 0)
                    / total_weight
                )
                diff = vec - oracle_client_vec
                Vc += weight * torch.dot(diff, diff).item()
            variance_client_g = Vc
            oracle_client_norm_sq = torch.dot(
                oracle_client_vec, oracle_client_vec
            ).item()
            variance_client_g_rel = (
                Vc / oracle_client_norm_sq
                if oracle_client_norm_sq > 0
                else float("nan")
            )

    variance_server_g = float("nan")
    variance_server_g_rel = float("nan")
    if USE_VARIANCE_G and server_vecs:
        oracle_server_vec = flatten_grad_list(g_manager.oracle_grads["server"])
        total_weight = sum(g_measure_state["server_batch_sizes"])
        if total_weight > 0:
            Vs = 0.0
            for vec, batch_size in zip(
                server_vecs, g_measure_state["server_batch_sizes"]
            ):
                weight = batch_size / total_weight
                diff = vec - oracle_server_vec
                Vs += weight * torch.dot(diff, diff).item()
            variance_server_g = Vs
            oracle_server_norm_sq = torch.dot(
                oracle_server_vec, oracle_server_vec
            ).item()
            variance_server_g_rel = (
                Vs / oracle_server_norm_sq
                if oracle_server_norm_sq > 0
                else float("nan")
            )

    split_g = float("nan")
    if (
        g_measure_state["client_split_grads"]
        and g_manager.oracle_grads.get("split") is not None
    ):
        split_stack = torch.stack(list(g_measure_state["client_split_grads"].values()))
        split_avg = split_stack.mean(dim=0)
        split_g = compute_g_score(g_manager.oracle_grads["split"], split_avg)

    client_sizes = [
        g_measure_state["client_batch_sizes"].get(cid, 0)
        for cid in g_measure_state["client_order"]
    ]
    server_sizes = list(g_measure_state["server_batch_sizes"])
    print(f"[G] Batch Sizes: client={client_sizes}, server={server_sizes}")

    for cid, gd in per_client_g.items():
        print(
            f"[G] Client {cid}: G={gd['G']:.6f}, G_rel={gd['G_rel']:.4f}, "
            f"D={gd['D_cosine']:.4f}"
        )
    if per_client_g:
        print(
            f"[G] Client Summary: G={avg_client_g:.6f}, G_rel={avg_g_rel:.4f}, "
            f"D={avg_d_cosine:.4f}"
        )

    for idx, gd in enumerate(server_g_list):
        batch_size = server_sizes[idx] if idx < len(server_sizes) else 0
        print(
            f"[G] Server {idx}: G={gd['G']:.6f}, G_rel={gd['G_rel']:.4f}, "
            f"D={gd['D_cosine']:.4f} (batch_size={batch_size})"
        )
    if server_g_list:
        print(
            f"[G] Server Summary: G={avg_server_g:.6f}, G_rel={avg_server_g_rel:.4f}, "
            f"D={avg_server_d:.4f}"
        )

    if USE_VARIANCE_G:
        print(
            f"[G] Variance Client: G={variance_client_g:.6f}, "
            f"G_rel={variance_client_g_rel:.6f}"
        )
        print(
            f"[G] Variance Server: G={variance_server_g:.6f}, "
            f"G_rel={variance_server_g_rel:.6f}"
        )

    per_client_g_payload = {
        str(cid): {
            "G": gd["G"],
            "G_rel": gd["G_rel"],
            "D": gd["D_cosine"],
        }
        for cid, gd in per_client_g.items()
    }
    per_server_g_payload = [
        {"G": gd["G"], "G_rel": gd["G_rel"], "D": gd["D_cosine"]}
        for gd in server_g_list
    ]

    if USE_VARIANCE_G:
        g_manager.g_history["client_g"].append(variance_client_g)
        g_manager.g_history["client_g_rel"].append(variance_client_g_rel)
        g_manager.g_history["client_d"].append(avg_d_cosine)
        g_manager.g_history["server_g"].append(variance_server_g)
        g_manager.g_history["server_g_rel"].append(variance_server_g_rel)
        g_manager.g_history["server_d"].append(avg_server_d)
    else:
        g_manager.g_history["client_g"].append(avg_client_g)
        g_manager.g_history["client_g_rel"].append(avg_g_rel)
        g_manager.g_history["client_d"].append(avg_d_cosine)
        g_manager.g_history["server_g"].append(avg_server_g)
        g_manager.g_history["server_g_rel"].append(avg_server_g_rel)
        g_manager.g_history["server_d"].append(avg_server_d)

    g_manager.g_history["variance_client_g"].append(variance_client_g)
    g_manager.g_history["variance_client_g_rel"].append(variance_client_g_rel)
    g_manager.g_history["variance_server_g"].append(variance_server_g)
    g_manager.g_history["variance_server_g_rel"].append(variance_server_g_rel)
    g_manager.g_history["split_g"].append(split_g)
    g_manager.g_history["per_client_g"].append(per_client_g_payload)
    g_manager.g_history["per_server_g"].append(per_server_g_payload)

    g_manager.oracle_grads = None
    g_measure_state["active"] = False
    g_measure_state["epoch"] = None
    g_measure_state["client_order"] = []
    g_measure_state["client_grads"].clear()
    g_measure_state["client_split_grads"].clear()
    g_measure_state["client_batch_sizes"].clear()
    g_measure_state["server_grads"].clear()
    g_measure_state["server_batch_sizes"].clear()
    # Clear accumulated/k_batch mode fields
    g_measure_state["accumulated_client_grads"].clear()
    g_measure_state["accumulated_client_samples"].clear()
    g_measure_state["accumulated_client_batch_counts"].clear()
    g_measure_state["accumulated_server_grads"] = None
    g_measure_state["accumulated_server_samples"] = 0
    g_measure_state["accumulated_server_batch_count"] = 0

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

_drift_epoch_started = False  # Track if drift measurement started for this epoch

# Pre-define dataset name for intermediate saving
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

# Pre-define config and filename for intermediate saving
_intermediate_timestamp = train_begin_time.strftime("%Y%m%d_%H%M%S")
_intermediate_json_filename = f"results/results_gas_{selectDataset}_{_intermediate_timestamp}.json"
os.makedirs("results", exist_ok=True)

_intermediate_config = {
    "dataset": selectDataset,
    "model": "ResNet-18" if use_resnet else "AlexNet",
    "split_layer": split_layer if use_resnet else None,
    "split_ratio": split_ratio if use_resnet else None,
    "split_alexnet": split_alexnet if not use_resnet else None,
    "epochs": epochs,
    "local_epochs": localEpoch,
    "users": user_num,
    "participating": user_parti_num,
    "batch_size": batchSize,
    "lr": lr,
    "momentum": momentum,
    "weight_decay": weight_decay,
    "clip_grad": clip_grad,
    "iid": iid,
    "dirichlet": dirichlet,
    "label_dirichlet": label_dirichlet,
    "alpha": alpha,
    "shard": shard,
    "min_require_size": min_require_size,
    "generate": Generate,
    "sample_frequency": Sample_Frequency,
    "v_test": V_Test,
    "v_test_frequency": V_Test_Frequency,
    "g_measurement": G_Measurement,
    "g_measure_frequency": G_Measure_Frequency,
    "seed": seed_value,
    "method": "Generative Activation-Aided" if Generate else "Original",
}


def _save_intermediate_results():
    """Save intermediate results to JSON (overwrites each round)"""
    results = {
        "config": _intermediate_config,
        "current_epoch": epoch,
        "total_epochs": epochs,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_record": time_record,
        "accuracy": total_accuracy,
        "v_value": total_v_value,
        "status": "in_progress" if epoch < epochs else "completed",
    }
    if G_Measurement and g_manager is not None:
        results["g_history"] = g_manager.get_history()
    if DRIFT_MEASUREMENT and drift_tracker is not None:
        results["drift_history"] = drift_tracker.get_history()
    with open(_intermediate_json_filename, "w") as f_json:
        json.dump(results, f_json, indent=4)


while epoch != epochs:
    user_model.train()
    server_model.train()

    # Drift Measurement: Start of epoch snapshot (client + server)
    if drift_tracker is not None and not _drift_epoch_started:
        drift_tracker.on_round_start(userParam, server_model.state_dict())
        _drift_epoch_started = True

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
    bn_states = None
    if labels.size(0) == 1:
        bn_states = _set_batchnorm_eval(server_model)
    local_output = server_model(split_layer_output)
    if bn_states is not None:
        _restore_batchnorm(server_model, bn_states)

    # localLoss = criterion(local_output, labels)
    localLoss = criterion(
        local_output + logit_local_adjustments[selected_client], labels.long()
    )
    optimizer_down.zero_grad()
    localLoss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(parameters=user_model.parameters(), max_norm=10)

    if g_measure_state is not None and g_measure_state["active"]:
        batch_size = labels.size(0)
        if G_Measurement_Accumulation in ("accumulated", "k_batch"):
            # Accumulated/K-batch mode: accumulate gradients for each client
            # K-batch mode: stop after K batches per client
            client_batch_count = g_measure_state["accumulated_client_batch_counts"].get(selected_client, 0)
            should_collect = True
            if G_Measurement_Accumulation == "k_batch" and client_batch_count >= G_Measurement_K:
                should_collect = False

            if should_collect:
                current_grads = [
                    p.grad.clone().cpu() * batch_size  # weight by batch size (grad_sum)
                    if p.grad is not None
                    else torch.zeros_like(p).cpu()
                    for p in user_model.parameters()
                ]
                if selected_client not in g_measure_state["accumulated_client_grads"]:
                    g_measure_state["client_order"].append(selected_client)
                    g_measure_state["accumulated_client_grads"][selected_client] = current_grads
                    g_measure_state["accumulated_client_samples"][selected_client] = batch_size
                    g_measure_state["accumulated_client_batch_counts"][selected_client] = 1
                else:
                    # Add to existing accumulated gradients
                    for i, grad in enumerate(current_grads):
                        g_measure_state["accumulated_client_grads"][selected_client][i] += grad
                    g_measure_state["accumulated_client_samples"][selected_client] += batch_size
                    g_measure_state["accumulated_client_batch_counts"][selected_client] += 1
                # Also collect split grad (only first batch for split layer)
                if (
                    selected_client not in g_measure_state["client_split_grads"]
                    and activation.grad is not None
                ):
                    g_measure_state["client_split_grads"][selected_client] = (
                        activation.grad.mean(dim=0).clone().cpu()
                    )
        else:
            # Single mode: only first batch per client
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
                g_measure_state["client_batch_sizes"][selected_client] = batch_size
                if activation.grad is not None:
                    g_measure_state["client_split_grads"][selected_client] = (
                        activation.grad.mean(dim=0).clone().cpu()
                    )
        finalize_g_measurement(g_measure_state, g_manager, user_parti_num)

    optimizer_down.step()
    usersParam[np.where(order == selected_client)[0][0]] = copy.deepcopy(
        user_model.state_dict()
    )

    # Drift Measurement: Accumulate drift after optimizer step
    if drift_tracker is not None:
        drift_tracker.accumulate_client_drift(
            selected_client, user_model.state_dict()
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
            server_batch_size = concat_labels.size(0)
            if G_Measurement_Accumulation in ("accumulated", "k_batch"):
                # Accumulated/K-batch mode: accumulate server gradients
                # K-batch mode: stop after K batches
                should_collect = True
                if G_Measurement_Accumulation == "k_batch":
                    if g_measure_state["accumulated_server_batch_count"] >= G_Measurement_K:
                        should_collect = False

                if should_collect:
                    current_grads = [
                        p.grad.clone().cpu() * server_batch_size  # weight by batch size
                        if p.grad is not None
                        else torch.zeros_like(p).cpu()
                        for p in server_model.parameters()
                    ]
                    if g_measure_state["accumulated_server_grads"] is None:
                        g_measure_state["accumulated_server_grads"] = current_grads
                        g_measure_state["accumulated_server_samples"] = server_batch_size
                        g_measure_state["accumulated_server_batch_count"] = 1
                    else:
                        for i, grad in enumerate(current_grads):
                            g_measure_state["accumulated_server_grads"][i] += grad
                        g_measure_state["accumulated_server_samples"] += server_batch_size
                        g_measure_state["accumulated_server_batch_count"] += 1
            else:
                # Single mode: only first batch
                if len(g_measure_state["server_grads"]) < 1:
                    g_measure_state["server_grads"].append(
                        [
                            p.grad.clone().cpu()
                            if p.grad is not None
                            else torch.zeros_like(p).cpu()
                            for p in server_model.parameters()
                        ]
                    )
                    g_measure_state["server_batch_sizes"].append(server_batch_size)
            finalize_g_measurement(g_measure_state, g_manager, user_parti_num)

        optimizer_up.step()

        # Drift Measurement: Accumulate server drift after optimizer step
        if drift_tracker is not None:
            drift_tracker.accumulate_server_drift(server_model.state_dict())

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
        # Drift Measurement: Finalize client when it completes local training
        if drift_tracker is not None:
            drift_tracker.finalize_client(
                selected_client,
                usersParam[np.where(order == selected_client)[0][0]]
            )
            drift_tracker.collect_client_delta(
                selected_client,
                usersParam[np.where(order == selected_client)[0][0]]
            )

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

            # Drift Measurement: Finalize round (client + server)
            if drift_tracker is not None:
                drift_tracker.on_round_end(epoch, userParam, server_model.state_dict())
                _drift_epoch_started = False  # Reset for next epoch

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
                    user_model,
                    server_model,
                    full_train_loader,
                    criterion,
                    full_model=full_model,
                    split_layer_name=split_layer,
                    use_sfl_oracle=USE_SFL_ORACLE,
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

                if g_manager.oracle_grads is not None:
                    oracle_client = g_manager.oracle_grads["client"]
                    oracle_server = g_manager.oracle_grads["server"]
                    split_grad = g_manager.oracle_grads.get("split")
                    client_vec = torch.cat([g.flatten().float() for g in oracle_client])
                    server_vec = torch.cat([g.flatten().float() for g in oracle_server])
                    split_shape = "none"
                    if split_grad is not None:
                        split_shape = [split_grad.shape]
                    total_samples = len(full_train_loader.dataset)
                    num_batches = len(full_train_loader)
                    print(
                        f"[G] Oracle: samples={total_samples}, batches={num_batches}, "
                        f"split_layer={split_layer}, split_shape={split_shape}"
                    )
                    print(
                        f"[G] Oracle Norms: client={torch.norm(client_vec).item():.4f} "
                        f"(numel={client_vec.numel()}), server={torch.norm(server_vec).item():.4f} "
                        f"(numel={server_vec.numel()})"
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

            # Save intermediate results after each round
            _save_intermediate_results()

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

# Save final results to JSON (same file as intermediate, with completed status)
results = {
    "config": _intermediate_config,
    "current_epoch": epoch,
    "total_epochs": epochs,
    "timestamp": end_time_str,
    "time_record": time_record,
    "accuracy": total_accuracy,
    "v_value": total_v_value,
    "status": "completed",
}
if G_Measurement and g_manager is not None:
    results["g_history"] = g_manager.get_history()

if DRIFT_MEASUREMENT and drift_tracker is not None:
    results["drift_history"] = drift_tracker.get_history()
    print(f"\n[Drift Measurement] Final G_drift values: {results['drift_history']['G_drift'][-5:]}")

with open(_intermediate_json_filename, "w") as f_json:
    json.dump(results, f_json, indent=4)

print(f"\nResults saved to {_intermediate_json_filename}")
