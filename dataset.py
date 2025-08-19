import os, torch, torchaudio, random
from torch.utils.data import Dataset

class InsectDataset(Dataset):
    def __init__(self, root_dir, mode='train', sample_rate=44100, duration=4.68):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)
        self.data, self.labels, self.label_map = [], [], {}
        path = os.path.join(root_dir, mode)
        for idx, folder in enumerate(sorted(os.listdir(path))):
            self.label_map[folder] = idx
            for file in os.listdir(os.path.join(path, folder)):
                if file.endswith(".wav"):
                    self.data.append(os.path.join(path, folder, file))
                    self.labels.append(idx)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.data[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
        if waveform.shape[1] < self.target_len:
            repeat = self.target_len // waveform.shape[1] + 1
            waveform = waveform.repeat(1, repeat)[:, :self.target_len]
        else:
            start = random.randint(0, waveform.shape[1] - self.target_len)
            waveform = waveform[:, start:start + self.target_len]
        return waveform, self.labels[idx]
