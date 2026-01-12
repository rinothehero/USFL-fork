import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


def compare_model_parameters(model_a, model_b):
    """
    두 PyTorch 모델(동일 구조 가정)의 state_dict를 비교하여
    각 파라미터(레이어)별 차이를 정량화해주는 함수입니다.

    Args:
        model_a (torch.nn.Module): 첫 번째 모델
        model_b (torch.nn.Module): 두 번째 모델

    Returns:
        diff_dict (dict):
            - key: 파라미터(레이어) 이름(str)
            - value: (l2_norm, mean_abs_diff) 튜플
    """
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    diff_dict = {}

    for key in sd_a.keys():
        print(key)
        param_a = sd_a[key]
        param_b = sd_b[key]
        diff = param_a.float() - param_b.float()
        l2_norm = torch.norm(diff, p=2).item()
        mean_abs_diff = diff.abs().mean().item()

        diff_dict[key] = (l2_norm, mean_abs_diff)

    return diff_dict


def evaluate(model, dataloader, device="mps"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def layerwise_swap_and_evaluate(model_a, model_b, test_loader, device="cuda"):
    # baseline 정확도 (아무것도 교체 안 한, model_a 그대로)
    base_acc = evaluate(model_a, test_loader, device)
    print(f"base_acc: {base_acc}")

    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    results = {}
    for key in sd_a.keys():
        print(key)
        # 임시 모델: model_a를 deepcopy한 뒤, 해당 레이어만 model_b 파라미터로 교체
        start_time = time.time()
        temp_model = copy.deepcopy(model_a)
        copy_time = time.time() - start_time
        print(f"Model deepcopy time: {copy_time:.4f}s")

        start_time = time.time()
        temp_sd = temp_model.state_dict()
        state_dict_time = time.time() - start_time
        print(f"Get state_dict time: {state_dict_time:.4f}s")

        # model_b state_dict에 해당 key가 없으면 패스(혹은 continue)
        if key not in sd_b:
            continue

        # 특정 키만 교체
        start_time = time.time()
        temp_sd[key] = sd_b[key]
        swap_time = time.time() - start_time
        print(f"Parameter swap time: {swap_time:.4f}s")

        # 수정된 state_dict 로드
        start_time = time.time()
        temp_model.load_state_dict(temp_sd)
        load_time = time.time() - start_time
        print(f"Load state_dict time: {load_time:.4f}s")

        # 테스트 정확도 측정
        start_time = time.time()
        swapped_acc = evaluate(temp_model, test_loader, device)
        eval_time = time.time() - start_time
        print(f"Evaluation time: {eval_time:.4f}s")

        # 정확도 차이
        diff_acc = swapped_acc - base_acc

        print(
            {
                "base_acc": base_acc,
                "swapped_acc": swapped_acc,
                "diff_acc": diff_acc,
            }
        )
        results[key] = {
            "base_acc": base_acc,
            "swapped_acc": swapped_acc,
            "diff_acc": diff_acc,
        }
    return results


def main():
    device = "cuda"

    # 하이퍼파라미터
    batch_size = 256

    # CIFAR-10 데이터셋: 학습은 이미 끝났다고 가정, 여기서는 테스트셋만 쓰면 됨
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model_a_ckpt_path = "cl.pth"
    model_b_ckpt_path = "usfl-2-0.1.pth"

    model_a = torch.load(
        model_a_ckpt_path,
    ).to(device)
    model_b = torch.load(
        model_b_ckpt_path,
    ).to(device)

    # 레이어 스와핑 후 평가
    results = layerwise_swap_and_evaluate(model_a, model_b, test_loader, device=device)

    # 출력
    for layer_name, info in results.items():
        print(
            f"[Layer: {layer_name}] "
            f"Base Acc: {info['base_acc']:.2f}% -> Swapped Acc: {info['swapped_acc']:.2f}% "
            f"(Diff: {info['diff_acc']:.2f}%)"
        )


if __name__ == "__main__":
    main()
