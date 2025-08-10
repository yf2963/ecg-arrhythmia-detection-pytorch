import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import signal
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from helper_code import *
from model_code import EnhancedNN
from sklearn.metrics import classification_report


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === CLASS MAPPINGS ===
# Removed AFib ('164889003'), keeping only AFlutter and PVC
CLASSES = ['164890007', '427172004']  # AFlutter, PVC

# === DATASET CLASS ===
class EnhancedECGDataset(Dataset):
    def __init__(self, header_files):
        self.files = header_files
        self.classes = CLASSES
        self.b, self.a = signal.butter(3, [1/250, 47/250], btype='band')

    def __len__(self):
        return len(self.files)

    def detect_r_peaks(self, lead_data):
        """Detect R peaks in Lead II for beat segmentation."""
        # Normalize data for peak detection
        normalized = lead_data / np.max(np.abs(lead_data)) if np.max(np.abs(lead_data)) > 0 else lead_data
        
        # Find R peaks - adjust parameters as needed for your data
        r_peaks, _ = find_peaks(np.abs(normalized), height=0.5, distance=150)
        
        # If too few peaks detected, try more sensitive parameters
        if len(r_peaks) < 3:
            r_peaks, _ = find_peaks(np.abs(normalized), height=0.3, distance=100)
        
        return r_peaks

    def __getitem__(self, idx):
        header_file = self.files[idx]
        record_file = header_file.replace('.hea', '.mat')
        header = load_header(header_file)
        recording = load_recording(record_file)

        # Preprocessing
        recording = signal.filtfilt(self.b, self.a, recording, axis=1)
        recording = (recording - np.mean(recording, axis=1, keepdims=True)) / (np.std(recording, axis=1, keepdims=True) + 1e-6)
        recording = np.nan_to_num(recording)

        # Beat segmentation - focus on Lead II for R peak detection
        lead_II = recording[1, :]  # Lead II is at index 1
        r_peaks = self.detect_r_peaks(lead_II)
        
        # Extract beat information
        beat_locations = np.zeros(10)  # Store up to 10 beat locations
        for i in range(min(10, len(r_peaks))):
            beat_locations[i] = r_peaks[i] / recording.shape[1]  # Normalize location
        
        # Zero-pad to 8192
        padded = np.zeros((12, 8192))  # 12-channel input to match model
        padded[:3, -recording.shape[1]:] = recording  # Place 3-lead into first 3 channels

        # Prepare labels
        labels = np.zeros(len(self.classes))
        dx = get_labels(header)
        for d in dx:
            if d in self.classes:
                labels[self.classes.index(d)] = 1
            # Map additional PVC code
            elif d == '17338001':  # Additional PVC code
                labels[self.classes.index('427172004')] = 1
            # Skip AFib labels
            elif d == '164889003':  # AFib code
                continue

        leads_present = np.zeros(12)
        leads_present[:3] = 1

        return (torch.tensor(padded, dtype=torch.float32), 
                torch.tensor(labels, dtype=torch.float32), 
                torch.tensor(leads_present, dtype=torch.float32),
                torch.tensor(beat_locations, dtype=torch.float32))

# === COLLATE ===
def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])  # [B, 12, 8192]
    t = torch.stack([item[1] for item in batch])  # [B, 2] - now 2 classes
    l = torch.stack([item[2] for item in batch])  # [B, 12]
    b = torch.stack([item[3] for item in batch])  # [B, 10] beat locations
    x = x.unsqueeze(2)  # [B, 12, 1, 8192]
    return x, t, l, b

# === TRAINING ===
def training_code(data_dir, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_dir = os.path.join(data_dir, 'reduced_train')
    val_dir = os.path.join(data_dir, 'reduced_val')

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.hea')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.hea')]

    train_dataset = EnhancedECGDataset(train_files)
    val_dataset = EnhancedECGDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Note: EnhancedNN will now get nOUT=2 instead of 3
    model = EnhancedNN(nOUT=len(CLASSES)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0.0
    best_epoch = 0
    patience = 5  # Early stopping patience
    no_improve = 0
    
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        for x, t, l, b in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, t, l, b = x.to(DEVICE), t.to(DEVICE), l.to(DEVICE), b.to(DEVICE)
            optimizer.zero_grad()
            logits, probs = model(x, l, b)
            loss = criterion(logits, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} - Training Loss: {running_loss / len(train_loader):.4f}")

        # Evaluation loop with per-class metrics
        model.eval()
        y_true = []
        y_pred = []
        y_scores = []

        with torch.no_grad():
            for x, t, l, b in val_loader:
                x, t, l, b = x.to(DEVICE), t.to(DEVICE), l.to(DEVICE), b.to(DEVICE)
                logits, probs = model(x, l, b)
                preds = (probs > 0.5).int().cpu().numpy()
                t = t.cpu().numpy()
                
                y_true.extend(t)
                y_pred.extend(preds)
                y_scores.extend(probs.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, target_names=CLASSES, 
                                      output_dict=True, zero_division=0)
        
        # Calculate average F1 - consider macro average for balanced metric
        avg_f1 = sum([report[cls]['f1-score'] for cls in CLASSES]) / len(CLASSES)
        print(f"Epoch {epoch} - Average F1: {avg_f1:.4f}")
        
        print("\n--- Per-Class Validation Metrics ---")
        print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
        
        # Save if best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch
            no_improve = 0
            print(f"New best model with F1: {best_f1:.4f}")
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs (best F1: {best_f1:.4f} at epoch {best_epoch})")
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))

# === INFERENCE ===
def load_model(model_dir, leads):
    model = EnhancedNN(nOUT=len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt'), map_location='cpu'))
    model.eval()
    return {'classifier': model, 'classes': CLASSES, 'thresholds': [0.5]*len(CLASSES)}

def run_model(model_dict, header, recording):
    input_leads = get_leads(header)
    full_input = np.zeros((12, recording.shape[1]))
    for i, lead in enumerate(input_leads):
        if lead in ['I', 'II', 'V2']:
            idx = ['I', 'II', 'V2'].index(lead)
            full_input[idx, :] = recording[i, :]

    # Preprocessing
    b, a = signal.butter(3, [1/250, 47/250], btype='band')
    full_input = signal.filtfilt(b, a, full_input, axis=1)
    full_input = (full_input - np.mean(full_input, axis=1, keepdims=True)) / (np.std(full_input, axis=1, keepdims=True) + 1e-6)
    full_input = np.nan_to_num(full_input)
    
    # Beat segmentation on Lead II
    lead_II = full_input[1, :]
    
    # Detect R peaks for beat segmentation
    normalized = lead_II / np.max(np.abs(lead_II)) if np.max(np.abs(lead_II)) > 0 else lead_II
    r_peaks, _ = find_peaks(np.abs(normalized), height=0.5, distance=150)
    if len(r_peaks) < 3:
        r_peaks, _ = find_peaks(np.abs(normalized), height=0.3, distance=100)
    
    # Extract beat information
    beat_locations = np.zeros(10)
    for i in range(min(10, len(r_peaks))):
        beat_locations[i] = r_peaks[i] / full_input.shape[1]
    
    # Zero padding
    padded = np.zeros((12, 8192))
    padded[:3, -full_input.shape[1]:] = full_input

    x = torch.tensor(padded).unsqueeze(0).unsqueeze(2).float()
    l = torch.tensor([1, 1, 1] + [0]*9).unsqueeze(0).float()
    b = torch.tensor(beat_locations).unsqueeze(0).float()

    model = model_dict['classifier']
    thresholds = model_dict['thresholds']
    classes = model_dict['classes']

    with torch.no_grad():
        logits, probs = model(x, l, b)
        probs = probs.cpu().numpy()[0]
        labels = (probs >= np.array(thresholds)).astype(int)

    return classes, labels, probs