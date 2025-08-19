'''
# train.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from thop import profile

from optimizer import get_optimizer_and_scheduler
from loss import get_loss_function


def train_model(model, train_loader, test_loader, num_classes, device, config, exp_name):
    base_dir = "run"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    log_file = open(os.path.join(exp_dir, "logs", "train.log"), "w")

    criterion = get_loss_function(config["loss"], num_classes)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, len(train_loader))

    best_acc, best_model, wait = 0, None, 0
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    epochs, patience = 100, 20

    for epoch in range(epochs):
        if epoch > 0:
            print(f"✅ Epoch {epoch}: "
                  f"Train Acc={train_accs[-1]:.2f}%, Test Acc={test_accs[-1]:.2f}%, "
                  f"Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}")
            log_file.write(f"Epoch {epoch}: "
                           f"Train Acc={train_accs[-1]:.2f}%, Test Acc={test_accs[-1]:.2f}%, "
                           f"Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}\n")
        model.train()
        total, correct, total_loss = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # 安全调用 scheduler.step()，避免 OneCycleLR 超步数报错
            #if scheduler: scheduler.step()
            if scheduler is not None:
                if hasattr(scheduler, "total_steps") and hasattr(scheduler, "_step_count"):
                    if scheduler._step_count < scheduler.total_steps:
                        scheduler.step()
                else:
                    # 对于非 OneCycleLR 的调度器（如 StepLR、CosineAnnealing 等）
                    scheduler.step()

            pred = out.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0); total_loss += loss.item()
        train_acc = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)
        train_accs.append(train_acc); train_losses.append(avg_train_loss)

        # eval
        model.eval(); correct, total, total_loss = 0, 0, 0
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                pred = out.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_score.extend(torch.softmax(out, dim=1).cpu().numpy())

        test_acc = 100 * correct / total
        avg_test_loss = total_loss / len(test_loader)
        test_accs.append(test_acc); test_losses.append(avg_test_loss)

        if (test_acc - best_acc)>0.01:
            best_acc = test_acc
            best_model = model.state_dict()
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if best_acc > 0.88:
                if (wait-20) >= patience:
                    print("⏹ Early stopping"); log_file.write("Early stopping\n")
                    break
            else:
                if wait >= patience:
                    print("⏹ Early stopping"); log_file.write("Early stopping\n")
                    break

    print(f"🏆 Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    log_file.write(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})\n")
    log_file.close()

    if best_model:
        torch.save(best_model, os.path.join(exp_dir, "models", "best_model.pth"))

    # ======== 指标评估 ========
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_score_np = np.array(y_score)
    per_class_ap = [average_precision_score(y_true_bin[:, i], y_score_np[:, i]) for i in range(num_classes)]
    mean_ap = np.mean(per_class_ap)
    flops, params = profile(model, inputs=(torch.randn(8, 1, 20635).to(device),), verbose=False)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "roc_auc_macro": roc_auc_score(y_true_bin, y_score_np, average='macro', multi_class='ovr'),
        "mean_ap": mean_ap,
        "per_class_ap": {f"class_{i}": float(ap) for i, ap in enumerate(per_class_ap)},
        "params_million": round(params / 1e6, 2),
        "flops_gflops": round(flops / 1e9, 4),
        "best_test_accuracy": round(best_acc, 2),
        "best_epoch": best_epoch
    }

    json.dump(metrics, open(os.path.join(exp_dir, "metrics", "metrics.json"), "w"), indent=2)
    classification_report_path = os.path.join(exp_dir, "metrics", "classification_report.csv")
    import pandas as pd
    pd.DataFrame(class_report).transpose().to_csv(classification_report_path)

    # ======= 可视化 ========
    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Training Curve"); plt.grid(); plt.legend()
    plt.savefig(os.path.join(exp_dir, "plots", "train_curve.png")); plt.close()

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(exp_dir, "plots", "confusion_matrix.png")); plt.close()

    plt.bar(range(num_classes), f1_score(y_true, y_pred, average=None, zero_division=0))
    plt.title("Per-Class F1"); plt.xlabel("Class"); plt.ylabel("F1")
    plt.savefig(os.path.join(exp_dir, "plots", "f1_per_class.png")); plt.close()

    if num_classes <= 10:
        for mode, func in [("roc", roc_curve), ("pr", precision_recall_curve)]:
            plt.figure()
            for i in range(num_classes):
                x, y, _ = func(y_true_bin[:, i], y_score_np[:, i])
                plt.plot(x, y, label=f"Class {i}")
            plt.title(f"{mode.upper()} Curve"); plt.legend()
            plt.savefig(os.path.join(exp_dir, "plots", f"{mode}_curve.png")); plt.close()

    json.dump(config, open(os.path.join(exp_dir, "logs", "config.json"), "w"), indent=2)
'''
# train.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from thop import profile

from optimizer import get_optimizer_and_scheduler
from loss import get_loss_function


def train_model(model, train_loader, test_loader, num_classes, device, config, exp_name):
    base_dir = "run"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    log_file = open(os.path.join(exp_dir, "logs", "train.log"), "w")

    criterion = get_loss_function(config["loss"], num_classes)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, len(train_loader))

    best_acc, best_model, wait = 0, None, 0
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    epochs, patience = 100, 20

    best_y_true, best_y_pred, best_y_score = [], [], []

    for epoch in range(epochs):
        if epoch > 0:
            print(f"✅ Epoch {epoch}: "
                  f"Train Acc={train_accs[-1]:.2f}%, Test Acc={test_accs[-1]:.2f}%, "
                  f"Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}")
            log_file.write(f"Epoch {epoch}: "
                           f"Train Acc={train_accs[-1]:.2f}%, Test Acc={test_accs[-1]:.2f}%, "
                           f"Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}\n")
        model.train()
        total, correct, total_loss = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Train"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if scheduler is not None:
                if hasattr(scheduler, "total_steps") and hasattr(scheduler, "_step_count"):
                    if scheduler._step_count < scheduler.total_steps:
                        scheduler.step()
                else:
                    scheduler.step()

            pred = out.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0); total_loss += loss.item()
        train_acc = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)
        train_accs.append(train_acc); train_losses.append(avg_train_loss)

        model.eval(); correct, total, total_loss = 0, 0, 0
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                pred = out.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                y_score.extend(torch.softmax(out, dim=1).cpu().numpy())

        test_acc = 100 * correct / total
        avg_test_loss = total_loss / len(test_loader)
        test_accs.append(test_acc); test_losses.append(avg_test_loss)

        if (test_acc - best_acc)>0.01:
            best_acc = test_acc
            best_model = model.state_dict()
            best_epoch = epoch + 1
            best_y_true = y_true.copy()
            best_y_pred = y_pred.copy()
            best_y_score = y_score.copy()
            wait = 0
        else:
            wait += 1
            if best_acc > 0.88:
                if (wait-20) >= patience:
                    print("⏹ Early stopping"); log_file.write("Early stopping\n")
                    break
            else:
                if wait >= patience:
                    print("⏹ Early stopping"); log_file.write("Early stopping\n")
                    break

    print(f"🏆 Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    log_file.write(f"Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})\n")
    log_file.close()

    if best_model:
        torch.save(best_model, os.path.join(exp_dir, "models", "best_model.pth"))

    class_report = classification_report(best_y_true, best_y_pred, output_dict=True, zero_division=0)
    y_true_bin = label_binarize(best_y_true, classes=list(range(num_classes)))
    y_score_np = np.array(best_y_score)
    per_class_ap = [average_precision_score(y_true_bin[:, i], y_score_np[:, i]) for i in range(num_classes)]
    mean_ap = np.mean(per_class_ap)
    flops, params = profile(model, inputs=(torch.randn(4, 1, 20635).to(device),), verbose=False)

    metrics = {
        "accuracy": accuracy_score(best_y_true, best_y_pred),
        "precision_macro": precision_score(best_y_true, best_y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(best_y_true, best_y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(best_y_true, best_y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(best_y_true, best_y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(best_y_true, best_y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(best_y_true, best_y_pred, average='weighted', zero_division=0),
        "roc_auc_macro": roc_auc_score(y_true_bin, y_score_np, average='macro', multi_class='ovr'),
        "mean_ap": mean_ap,
        "per_class_ap": {f"class_{i}": float(ap) for i, ap in enumerate(per_class_ap)},
        "params_million": round(params / 1e6, 2),
        "flops_gflops": round(flops / 1e9, 4),
        "best_test_accuracy": round(best_acc, 2),
        "best_epoch": best_epoch
    }

    json.dump(metrics, open(os.path.join(exp_dir, "metrics", "metrics.json"), "w"), indent=2)
    classification_report_path = os.path.join(exp_dir, "metrics", "classification_report.csv")
    import pandas as pd
    pd.DataFrame(class_report).transpose().to_csv(classification_report_path)

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Training Curve"); plt.grid(); plt.legend()
    plt.savefig(os.path.join(exp_dir, "plots", "train_curve.png")); plt.close()

    cm = confusion_matrix(best_y_true, best_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(exp_dir, "plots", "confusion_matrix.png")); plt.close()

    plt.bar(range(num_classes), f1_score(best_y_true, best_y_pred, average=None, zero_division=0))
    plt.title("Per-Class F1"); plt.xlabel("Class"); plt.ylabel("F1")
    plt.savefig(os.path.join(exp_dir, "plots", "f1_per_class.png")); plt.close()

    if num_classes <= 10:
        for mode, func in [("roc", roc_curve), ("pr", precision_recall_curve)]:
            plt.figure()
            for i in range(num_classes):
                x, y, _ = func(y_true_bin[:, i], y_score_np[:, i])
                plt.plot(x, y, label=f"Class {i}")
            plt.title(f"{mode.upper()} Curve"); plt.legend()
            plt.savefig(os.path.join(exp_dir, "plots", f"{mode}_curve.png")); plt.close()

    json.dump(config, open(os.path.join(exp_dir, "logs", "config.json"), "w"), indent=2)
