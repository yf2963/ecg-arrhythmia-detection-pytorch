import os
import torch
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader
from helper_code import *
from model_code import EnhancedNN as NN

from scipy import signal
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['164890007', '427172004']  # AFlutter, PVC

class ECGDataset(Dataset):
    def __init__(self, header_files):
        self.files = header_files
        self.classes = CLASSES
        self.b, self.a = signal.butter(3, [1/250, 47/250], btype='band')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        header_file = self.files[idx]
        record_file = header_file.replace('.hea', '.mat')
        header = load_header(header_file)
        recording = load_recording(record_file)

        recording = signal.filtfilt(self.b, self.a, recording, axis=1)
        recording = (recording - np.mean(recording, axis=1, keepdims=True)) / (np.std(recording, axis=1, keepdims=True) + 1e-6)
        recording = np.nan_to_num(recording)

        padded = np.zeros((12, 8192))
        padded[:3, -recording.shape[1]:] = recording

        labels = np.zeros(len(self.classes))
        dx = get_labels(header)
        for d in dx:
            if d in self.classes:
                labels[self.classes.index(d)] = 1
            elif d == '17338001':  # Additional PVC synonym
                labels[self.classes.index('427172004')] = 1

        leads_present = np.zeros(12)
        leads_present[:3] = 1

        return torch.tensor(padded, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), torch.tensor(leads_present, dtype=torch.float32)

def evaluate_model(model_path, test_dir):
    print("\nEvaluating model on test set...")
    print(f"Classes: {CLASSES}")

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.hea')]
    test_dataset = ECGDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = NN(nOUT=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for x, t, l in test_loader:
            x, t, l = x.to(DEVICE), t.to(DEVICE), l.to(DEVICE)
            x = x.unsqueeze(2)  # [B, 12, 1, 8192]
            logits, probs = model(x, l)
            preds = (probs > 0.5).int().cpu().numpy()
            y_scores.extend(probs.cpu().numpy())
            y_pred.extend(preds)
            y_true.extend(t.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    print("\n--- Per-Class Test Metrics ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    print("\n--- Confusion Matrices, FNR, and FPR ---")
    for i, class_name in enumerate(CLASSES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\nClass: {class_name}")
        print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")

    print("\n--- Summary ---")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    print(f"\nAverage F1: {np.mean(f1):.4f}")

    plot_precision_recall(y_true, y_scores, CLASSES)

def plot_precision_recall(y_true, y_scores, class_names):
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = average_precision_score(y_true[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP={ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (AFlutter & PVC)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('aflutter_pvc_precision_recall_curves.png')
    print("\nSaved PR curve to 'aflutter_pvc_precision_recall_curves.png'")

evaluate_model("model_aflutter_pvc/best_model.pt", "reduced_test")


