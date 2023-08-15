import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target) # 예측 클래스와 타겟 클래스가 일치하는 경우 True로 채워진 불리언 텐서가 생성됩니다.
        correct = 0
        correct += torch.sum(pred == target).item() # 일치하는 총 개수를 계산합니다.
    return correct / len(target) # 정확한 예측의 비율을 계산하여 반환합니다.

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item() # 상위 k개 클래스 중 i번째 클래스와 타겟 클래스가 일치하는 경우 True로 채워진 불리언 텐서가 생성됩니다.
    return correct / len(target)