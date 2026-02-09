import torch
import numpy as np
from log_utils import vprint


def adjust_positive_definite(loc, matrix, device, max_attempts=100, epsilon_start=1e-6):
    """
    Adjust the matrix by gradually increasing the positive values on the diagonal until it becomes positive definite.
    """
    epsilon = epsilon_start
    for attempt in range(max_attempts):
        try:
            # Attempt to check positive definiteness using Cholesky decomposition.
            eye_matrix = epsilon * torch.eye(
                matrix.size(0), device=device
            )  # 在相同设备上创建单位矩阵
            mvn = torch.distributions.MultivariateNormal(loc, matrix + eye_matrix)
            return mvn, True
        except ValueError:
            epsilon *= 10  # If the Cholesky decomposition fails, increase the epsilon value and try again.
    return matrix, False


def try_multivariate_normal(loc, covariance_matrix, device):
    """
    Attempt to create a multivariate normal distribution; if the covariance matrix is not positive definite, automatically adjust it.
    """
    try:
        mvn = torch.distributions.MultivariateNormal(loc, covariance_matrix)
        return mvn
    except ValueError as e:
        if "PositiveDefinite" in str(e):
            vprint(
                "The covariance matrix is not positive definite, attempting to adjust it....", 0
            )
            mvn, success = adjust_positive_definite(loc, covariance_matrix, device)
            if success:
                return mvn
            else:
                raise ValueError(
                    "Unable to adjust the covariance matrix to make it positive definite."
                )
        else:
            raise


def _sample_diagonal_features(mean, variance, samples_needed, device):
    if variance.dim() >= 2:
        diag_var = torch.diagonal(variance, dim1=-2, dim2=-1)
    else:
        diag_var = variance
    std = torch.sqrt(diag_var + 1e-5)
    mean_expanded = mean.unsqueeze(0).expand(samples_needed, -1)
    std_expanded = std.unsqueeze(0).expand(samples_needed, -1)
    return torch.normal(mean=mean_expanded, std=std_expanded).to(device)


def calculate_v_value(
    server_model,
    user_model,
    concat_features,
    concat_labels,
    test_loader,
    criterion,
    device,
    num_minibatches=10,
):
    total_gradients = [torch.zeros_like(param) for param in server_model.parameters()]

    # Estimate the true gradient
    for _ in range(num_minibatches):
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)

        split_layer_output = user_model(images)
        server_output = server_model(split_layer_output)
        loss = criterion(server_output, labels.long())
        grads_server = torch.autograd.grad(
            loss, server_model.parameters(), retain_graph=True, allow_unused=True
        )
        for total_grad, grad in zip(total_gradients, grads_server):
            if grad is None:
                continue
            total_grad += grad
        server_model.zero_grad()
        user_model.zero_grad()

    # Calculate the average gradient
    grads_real = [total_grad / num_minibatches for total_grad in total_gradients]

    # Calculate gradient dissimilarity
    server_output = server_model(concat_features)
    loss = criterion(server_output, concat_labels.long())
    grads_sampled = torch.autograd.grad(
        loss,
        server_model.parameters(),
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )
    # Filter out None gradients (from unused parameters in flexible split)
    grads_sampled = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads_sampled, server_model.parameters())
    ]

    v_value = sum(
        (torch.norm(g - gr) ** 2).item() for g, gr in zip(grads_sampled, grads_real)
    ) / len(grads_sampled)

    return v_value


def replace_user(order, k, user_num):
    available_users = set(range(user_num)) - set(order)
    if len(available_users) == 0:
        # All clients are participating - reuse the same client (reset state)
        new_user = k
    else:
        new_user = np.random.choice(list(available_users))
    order[np.where(order == k)[0][0]] = new_user
    return order


def sample_or_generate_features(
    concat_features,
    concat_labels,
    batchsize,
    num_labels,
    original_shape,
    device,
    stats,
    diagonal=True,
):
    """
    Sample or generate features.
    Args:
        diagonal: If True (ResNet), use torch.normal with diagonal variance.
                  If False (AlexNet), use MultivariateNormal with full covariance.
    """
    new_features_list = []
    new_labels_list = []
    for label in range(num_labels):
        label_mask = concat_labels == label
        label_features = concat_features[label_mask]

        mean, variance = stats.get_stats(label)
        mean = mean.to(device)
        variance = variance.to(device)

        if label_features.size(0) > 0:
            if label_features.size(0) >= batchsize:
                sampled_features = label_features[:batchsize]
            else:  # If the activations are insufficient, generate more
                samples_needed = batchsize - label_features.size(0)
                if diagonal:
                    # Diagonal variance (ResNet)
                    std = torch.sqrt(variance + 1e-5)
                    mean_expanded = mean.unsqueeze(0).expand(samples_needed, -1)
                    std_expanded = std.unsqueeze(0).expand(samples_needed, -1)
                    generated_features = torch.normal(
                        mean=mean_expanded, std=std_expanded
                    ).to(device)
                else:
                    # Full covariance (AlexNet)
                    try:
                        mvn = try_multivariate_normal(mean, variance, device)
                        generated_features = mvn.sample(
                            torch.Size([samples_needed])
                        ).to(device)
                    except ValueError:
                        generated_features = _sample_diagonal_features(
                            mean, variance, samples_needed, device
                        )
                # Restore the sampled features back to original dimensions
                restored_features = generated_features.reshape(
                    samples_needed, *original_shape
                )
                sampled_features = torch.cat([label_features, restored_features], dim=0)
        else:
            # If the current label does not exist in concat_features, directly generate the activations.
            samples_needed = batchsize
            if diagonal:
                # Diagonal variance (ResNet)
                std = torch.sqrt(variance + 1e-5)
                mean_expanded = mean.unsqueeze(0).expand(samples_needed, -1)
                std_expanded = std.unsqueeze(0).expand(samples_needed, -1)
                generated_features = torch.normal(
                    mean=mean_expanded, std=std_expanded
                ).to(device)
            else:
                # Full covariance (AlexNet)
                try:
                    mvn = try_multivariate_normal(mean, variance, device)
                    generated_features = mvn.sample(torch.Size([samples_needed])).to(
                        device
                    )
                except ValueError:
                    generated_features = _sample_diagonal_features(
                        mean, variance, samples_needed, device
                    )
            sampled_features = generated_features.reshape(
                samples_needed, *original_shape
            )

        new_features_list.append(sampled_features)
        new_labels_list.append(
            torch.full(
                (sampled_features.size(0),), fill_value=label, dtype=concat_labels.dtype
            )
        )

    # Concatenate all processed activations and labels
    new_concat_features = torch.cat(new_features_list, dim=0).to(device)
    new_concat_labels = torch.cat(new_labels_list, dim=0).to(device)
    return new_concat_features, new_concat_labels


""" LA """


def compute_local_adjustment(train_loader, device, numOfLabel=10, tro=1):
    label_freq = {}
    for i in range(numOfLabel):
        label_freq[i] = 0
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    # print(label_freq)
    label_freq_array = np.array(list(label_freq.values()))
    # print(label_freq_array)
    label_freq_array = label_freq_array / label_freq_array.sum()
    # print(label_freq_array)
    adjustments = np.log(label_freq_array**tro + 1e-12)
    adjustments = torch.from_numpy(adjustments).float()
    adjustments = adjustments.to(device)
    return adjustments


def find_client_with_min_time(clients, order):
    # Initialize the minimum time and the corresponding client
    min_time = float("inf")
    client_with_min_time = None
    # Iterate over each index in order
    for index in order:
        client = clients[index]
        if client.time < min_time:
            min_time = client.time
            client_with_min_time = index

    return client_with_min_time
