import numpy as np
import jsonlines
import argparse
import math
from sklearn.metrics import f1_score

def sigmoid(score, k, tau):
    """Sigmoid transform: s = (1 + exp(-k*(score - tau)))^-1"""
    """k(float): slope
    factor.
    - If
    method is Fast - DetectGPT, set
    k = 1
    - If
    method is Binoculars, set
    k = -1


    tau(float): detector preset threshold, eg. Binoculars: 0.901"""
    return 1.0 / (1.0 + np.exp(-k * (score - tau)))
    # return score

def load_calibration_scores(file_path, k, tau, bucket_len, max_len=1000):
    """Load calibration scores and split into buckets."""
    cal_len = math.ceil(max_len / bucket_len)
    cal_scores = [[] for _ in range(cal_len)]

    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            text_len = item["length"]
            s = sigmoid(item["score"], k, tau)
            flag = text_len // bucket_len
            if flag >= cal_len - 1:
                cal_scores[cal_len - 1].append(s)
            else:
                cal_scores[flag].append(s)

    return cal_scores, cal_len

def compute_qhat(cal_scores, alpha):
    """Compute conformal qhat per bucket."""
    qhat_list = []
    for bucket in cal_scores:
        n = len(bucket)
        if n == 0:
            qhat_list.append(1.0)  # 如果没数据，给个保守阈值
            continue
        q = np.ceil((n + 1) * (1 - alpha)) / n
        q = min(max(0, q), 1)
        qhat_list.append(np.quantile(bucket, q))
    return qhat_list

def evaluate(file_path, qhat_list, k, tau, bucket_len, max_len=1000):
    """Evaluate predictions on a test file."""
    cal_len = len(qhat_list)
    human_preds, machine_preds = [], []

    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            text_len = item["length"]
            s = sigmoid(item["score"], k, tau)

            if text_len >= max_len - bucket_len:
                qhat = qhat_list[cal_len - 1]
            else:
                qhat = qhat_list[text_len // bucket_len]

            pred = s > qhat

            if item["label"] == "human":
                human_preds.append(pred)
            else:
                machine_preds.append(pred)

    return human_preds, machine_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_file", default="M4_fastdetectGPT_calibrate.jsonl", help="Calibration file (jsonl)")
    parser.add_argument("--human_test_file", default="M4_fastdetectGPT_human_test.jsonl", help="Human test file (jsonl)")
    parser.add_argument("--machine_test_file", default="M4_fastdetectGPT_machine_test.jsonl", help="Machine test file (jsonl)")
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--bucket_len", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--max_len", type=int, default=1000)
    args = parser.parse_args()

    # Step 1: calibration
    cal_scores, cal_len = load_calibration_scores(
        args.calib_file, args.k, args.tau, args.bucket_len, args.max_len
    )
    qhat_list = compute_qhat(cal_scores, args.alpha)

    # Step 2: evaluate human test
    human_preds1, machine_preds1 = evaluate(
        args.human_test_file, qhat_list, args.k, args.tau, args.bucket_len, args.max_len
    )

    # Step 3: evaluate machine test
    human_preds2, machine_preds2 = evaluate(
        args.machine_test_file, qhat_list, args.k, args.tau, args.bucket_len, args.max_len
    )

    # Step 4: combine
    human_preds = human_preds1 + human_preds2
    machine_preds = machine_preds1 + machine_preds2

    y_true = [0] * len(human_preds) + [1] * len(machine_preds)
    y_pred = [int(p) for p in human_preds] + [int(p) for p in machine_preds]

    fpr = np.mean(human_preds)  # FPR = human 被预测成 AI 的比例
    tpr = np.mean(machine_preds)  # TPR = machine 被预测成 AI 的比例
    f1 = f1_score(y_true, y_pred)

    print("FPR:", fpr)
    print("TPR:", tpr)
    print("F1 :", f1)
