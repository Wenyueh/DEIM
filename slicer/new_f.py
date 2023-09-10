import torch


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


def calculate_error_instances(ys, y_hats, slices):
    accuracies = []
    slice_accuracies = {}
    for index in range(ys.shape[0]):
        if ys[index][0] == 1 and y_hats[index][0] > y_hats[index][1]:
            accuracies.append(1)
        elif ys[index][1] == 1 and y_hats[index][1] > y_hats[index][0]:
            accuracies.append(1)
        else:
            accuracies.append(0)

        if slices[index] not in slice_accuracies:
            slice_accuracies[slices[index]] = {"accurate": 0, "total": 1e-6}
        if accuracies[-1] == 1:
            slice_accuracies[slices[index]]["accurate"] += 1
        slice_accuracies[slices[index]]["total"] += 1

    errors_found = 1e-6
    true_errors = 0
    num_error_cluster = 0
    active_clusters = 0
    for cluster in slice_accuracies:
        active_clusters += 1
        acc = slice_accuracies[cluster]["accurate"] / slice_accuracies[cluster]["total"]
        if acc < 0.5:
            num_error_cluster += 1
            errors_found += slice_accuracies[cluster]["total"]
            true_errors += (
                slice_accuracies[cluster]["total"]
                - slice_accuracies[cluster]["accurate"]
            )

    return true_errors / errors_found, errors_found, num_error_cluster, active_clusters
