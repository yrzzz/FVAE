
import os, glob, random, argparse, math
from collections import defaultdict

import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ------------------------  DATA  ------------------------
class RawWaveDataset(Dataset):
    def __init__(self, root, fixed_len=16_000):
        self.files      = sorted(glob.glob(os.path.join(root, '*', '*.wav')))
        self.fixed_len  = fixed_len
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        wav = torch.cat([wav, torch.zeros(1, max(0, self.fixed_len - wav.size(1)))], 1)[:, :self.fixed_len]
        return wav, idx

class AudioMNIST(Dataset):
    """Return log-Mel specs in [0,1] with shape [1,n_mels,frames]."""
    def __init__(self, root, transform, fixed_len=16_000, max_frames=128):
        self.files      = sorted(glob.glob(os.path.join(root, '*', '*.wav')))
        self.transform  = transform
        self.fixed_len  = fixed_len
        self.max_frames = max_frames
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        wav = torch.cat([wav, torch.zeros(1, max(0, self.fixed_len - wav.size(1)))], 1)[:, :self.fixed_len]
        spec = self.transform(wav)                     # [1,n_mels,T]
        Tm   = spec.size(2)
        if Tm < self.max_frames:
            pad = torch.zeros(1, spec.size(1), self.max_frames - Tm)
            spec = torch.cat([spec, pad], 2)
        else:
            spec = spec[:, :, : self.max_frames]
        label = int(os.path.basename(self.files[idx]).split('_')[0])
        return spec, label

class PairedMNISTAudio(Dataset):
    """Return (image, spectrogram) pairs sharing the digit label."""
    def __init__(self, mnist_ds, audio_ds):
        self.mnist, self.audio = mnist_ds, audio_ds
        self.by_lab = defaultdict(list)
        for i, (_, lab) in enumerate(self.mnist):
            self.by_lab[int(lab)].append(i)
    def __len__(self): return len(self.audio)
    def __getitem__(self, idx):
        spec, lab = self.audio[idx]
        img_idx   = random.choice(self.by_lab[int(lab)])
        img, _    = self.mnist[img_idx]
        return img, spec, lab

# ----------------------  MODELS  ------------------------
class ImageAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Flatten(), nn.Linear(64*7*7, z_dim))
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 64*7*7), nn.ReLU(True),
            nn.Unflatten(1,(64,7,7)),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(64,32,3,1,1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,1,3,1,1),  nn.Sigmoid())
    def encode(self,x): return self.enc(x)
    def decode(self,z): return self.dec(z)

class AudioAE(nn.Module):
    """Works for any (n_mels, max_frames)."""
    def __init__(self, z_dim=64, n_mels=64, max_frames=128):
        super().__init__()
        self.conv_body = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1,1,n_mels,max_frames)
            feat  = self.conv_body(dummy)             # [1,128,Hc,Wc]
            C, Hc, Wc = feat.shape[1], feat.shape[2], feat.shape[3]
            flat = C * Hc * Wc
        self._shapes = (C, Hc, Wc)
        self.enc_fc = nn.Linear(flat, z_dim)
        self.dec_fc = nn.Linear(z_dim, flat)
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(128,64,3,1,1), nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(64,32,3,1,1),  nn.ReLU(True),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,1,3,1,1),   nn.Sigmoid())
    def encode(self,x):
        x = self.conv_body(x)
        x = torch.flatten(x, 1)
        return self.enc_fc(x)
    def decode(self,z):
        x = self.dec_fc(z)
        C,Hc,Wc = self._shapes
        x = x.view(-1, C, Hc, Wc)
        return self.deconv(x)

# ----------------------  Enc-RF module  ----------------------
class TimeEmbed(nn.Module):
    """Tiny sinusoidal time embedding (dim=16)."""
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        # fixed frequencies
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dim//2))
        self.register_buffer("freqs", freqs, persistent=False)
    def forward(self, t):  # t: [B,1] in [0,1]
        ang = t * self.freqs[None, :] * 2*math.pi
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return emb  # [B, dim]

class EncRF(nn.Module):
    """Rectified-Flow head in latent space: predicts velocity v(z_t,t)."""
    def __init__(self, z_dim, t_dim=16, hidden=512):
        super().__init__()
        self.tok = TimeEmbed(t_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim + t_dim),
            nn.Linear(z_dim + t_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, z_dim)
        )
    def forward(self, z_t, t):
        # z_t: [B, z], t: [B, 1] in [0,1]
        te  = self.tok(t)            # [B, t_dim]
        inp = torch.cat([z_t, te], -1)
        return self.net(inp)         # [B, z_dim]

def rf_alignment_loss(rf: EncRF, zi: torch.Tensor, za: torch.Tensor):
    """
    Symmetric Enc-RF:
      sample t ~ U(0,1), z_t=(1-t)zi + t za, target (za-zi).
      also swap (za,zi) for symmetry and average two losses.
    """
    B, D = zi.size(0), zi.size(1)
    device = zi.device

    t = torch.rand(B, 1, device=device)
    zt_ia = (1.0 - t)*zi + t*za
    v_pred_ia = rf(zt_ia, t)
    v_tgt_ia  = (za - zi)

    # swap
    t2 = torch.rand(B, 1, device=device)
    zt_ai = (1.0 - t2)*za + t2*zi
    v_pred_ai = rf(zt_ai, t2)
    v_tgt_ai  = (zi - za)

    loss = F.mse_loss(v_pred_ia, v_tgt_ia) + F.mse_loss(v_pred_ai, v_tgt_ai)
    return 0.5 * loss

def symm_kl_gauss_diag(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-5):
    # z1,z2: [B, Z]
    mu1, mu2 = z1.mean(0), z2.mean(0)                      # [Z]
    var1 = z1.var(0, unbiased=False) + eps                 # [Z]
    var2 = z2.var(0, unbiased=False) + eps                 # [Z]
    # KL(N1||N2) for diagonal cov
    kl12 = 0.5 * (torch.log(var2/var1).sum()
                  + ((var1 + (mu1 - mu2)**2)/var2).sum()
                  - z1.size(1))
    kl21 = 0.5 * (torch.log(var1/var2).sum()
                  + ((var2 + (mu2 - mu1)**2)/var1).sum()
                  - z1.size(1))
    return 0.5*(kl12 + kl21)

# -------------------  TRAIN / EVAL  --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', type=int, default=64, help='shared embedding dim')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lambda_rf', type=float, default=0.5, help='weight for Enc-RF loss')
    parser.add_argument('--results', type=str, default='results_rf_min')
    parser.add_argument('--outdir', type=str, default='eval_rf_min')
    args, _ = parser.parse_known_args()

    BATCH, Z, EPOCHS = args.batch, args.z, args.epochs
    LAMBDA_RF = args.lambda_rf

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.outdir,  exist_ok=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- audio params / transforms
    mel_args = dict(
        sample_rate=16_000,
        n_fft=1024, hop_length=128, win_length=1024,
        n_mels=64, f_min=0.0, f_max=8000.0,
        mel_scale="htk", norm="slaney"
    )
    img_tf=T.ToTensor()
    aud_tf = T.Compose([
        torchaudio.transforms.MelSpectrogram(**mel_args),
        torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
        T.Lambda(lambda x: (x + 80) / 80)  # [0,1]
    ])

    # ---- data
    mnist_root='data'
    audio_root='data/AudioMNIST/data'
    mnist_tr = datasets.MNIST(mnist_root,True ,img_tf,download=True)
    audio_tr = AudioMNIST(audio_root,aud_tf,16_000,128)
    train_dl = DataLoader(PairedMNISTAudio(mnist_tr,audio_tr),
                          batch_size=BATCH,shuffle=True,num_workers=4)

    mnist_te = datasets.MNIST(mnist_root,False,img_tf,download=True)
    audio_te = AudioMNIST(audio_root,aud_tf,16_000,128)
    test_set = PairedMNISTAudio(mnist_te,audio_te)
    # test_dl  = DataLoader(test_set, batch_size=256, shuffle=False)

    # ---- models
    img_ae = ImageAE(Z).to(device)
    aud_ae = AudioAE(Z, n_mels=mel_args["n_mels"], max_frames=128).to(device)
    enc_rf = EncRF(Z).to(device)

    # ---- optimizers
    optim_all = optim.Adam(
        list(img_ae.parameters()) + list(aud_ae.parameters()) + list(enc_rf.parameters()),
        lr=1e-3
    )
    L2 = nn.MSELoss()

    # ---- train
    for ep in range(1, EPOCHS+1):
        img_ae.train(); aud_ae.train(); enc_rf.train()
        run=0.0; N=0
        for img, spec, _ in train_dl:
            img, spec = img.to(device), spec.to(device)

            # encodings
            zi = img_ae.encode(img)     # [B,Z]
            za = aud_ae.encode(spec)    # [B,Z]
            # recon losses (self-recon only)
            # loss_img = L2(img_ae.decode(zi), img,)
            # loss_aud = L2(aud_ae.decode(za), spec)
            # z_joint = 0.5 * (zi + za)  # optional: shared "center"
            # zi, za = z_joint, z_joint   # force identical embeddings
            loss_img = F.mse_loss(img_ae.decode(zi), img, reduction='mean')
            loss_aud = F.mse_loss(aud_ae.decode(za), spec, reduction='mean')
            loss_rec = loss_img + loss_aud

            # Enc-RF alignment on embeddings (symmetric)
            #loss_rf = rf_alignment_loss(enc_rf, zi, za)
            loss_rf = symm_kl_gauss_diag(zi, za)
            loss = loss_rec + LAMBDA_RF * loss_rf

            optim_all.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(img_ae.parameters()) + list(aud_ae.parameters()) + list(enc_rf.parameters()), 5.0)
            optim_all.step()

            bs = img.size(0)
            run += loss.item() * bs
            N   += bs

        print(f"Epoch {ep:02d}/{EPOCHS}  Loss {run/max(1,N):.4f}  (rec + {LAMBDA_RF}*encRF)")

    # ---- save
    torch.save(img_ae.state_dict(), os.path.join(args.results,'img_ae.pth'))
    torch.save(aud_ae.state_dict(), os.path.join(args.results,'aud_ae.pth'))
    torch.save(enc_rf.state_dict(), os.path.join(args.results,'enc_rf.pth'))

    # ---------------- evaluation (quick sanity) ----------------
    img_ae.eval(); aud_ae.eval(); enc_rf.eval()

    # ---------- qualitative samples ----------
    os.makedirs(args.outdir, exist_ok=True)
    idxs  = random.sample(range(len(test_set)), 10)
    imgs  = torch.stack([test_set[i][0] for i in idxs]).to(device)
    specs = torch.stack([test_set[i][1] for i in idxs]).to(device)

    with torch.no_grad():
        zi_s  = img_ae.encode(imgs)
        za_s  = aud_ae.encode(specs)

        img_self_s  = img_ae.decode(zi_s)
        img_from_aud= img_ae.decode(za_s)

        aud_self_spec01   = aud_ae.decode(za_s).squeeze(1)   # [B, n_mels, T] in [0,1]
        aud_from_img_spec = aud_ae.decode(zi_s).squeeze(1)   # [B, n_mels, T] in [0,1]

    # --- save images: orig / self / cross (aud->img)
    for k in range(len(idxs)):
        plt.imsave(f'{args.outdir}/img_{k}_orig.png',
                imgs[k].cpu().squeeze().numpy(), cmap='gray')
        plt.imsave(f'{args.outdir}/img_{k}_self.png',
                img_self_s[k].cpu().squeeze().numpy(), cmap='gray')
        plt.imsave(f'{args.outdir}/img_{k}_from_aud.png',
                img_from_aud[k].cpu().squeeze().numpy(), cmap='gray')

    # --- audio inversion helper (dB[power] -> power -> lin -> mag -> Griffin-Lim)
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=mel_args["n_fft"]//2 + 1,
        n_mels=mel_args["n_mels"],
        sample_rate=mel_args["sample_rate"],
        f_min=mel_args["f_min"], f_max=mel_args["f_max"],
        mel_scale=mel_args["mel_scale"], norm=mel_args["norm"]
    ).to(device)
    gl = torchaudio.transforms.GriffinLim(
        n_fft=mel_args["n_fft"], hop_length=mel_args["hop_length"], win_length=mel_args["win_length"],
        window_fn=torch.hann_window, power=1.0, n_iter=80
    ).to(device)

    def spec01_to_wav(spec01: torch.Tensor) -> torch.Tensor:
        # spec01: [B, n_mels, T] in [0,1]
        db   = spec01*80 - 80              # back to dB (power)
        melP = 10.0**(db/10.0)             # power
        linP = inv_mel(melP)               # linear power
        mag  = torch.sqrt(torch.clamp(linP, min=1e-10))
        return gl(mag)                     # [B, T]

    # --- save audio: orig(spec inversion) / self / cross (img->aud)
    wav_self   = spec01_to_wav(aud_self_spec01)               # audio -> audio
    wav_from_i = spec01_to_wav(aud_from_img_spec)             # image -> audio
    for k in range(len(idxs)):
        # optional: invert the original spec as reference (approximate)
        wav_orig = spec01_to_wav(specs.squeeze(1))[k].detach().unsqueeze(0).cpu()
        torchaudio.save(f'{args.outdir}/aud_{k}_orig_est.wav', wav_orig, 16_000)

        torchaudio.save(f'{args.outdir}/aud_{k}_self.wav',
                        wav_self[k].detach().unsqueeze(0).cpu(), 16_000)
        torchaudio.save(f'{args.outdir}/aud_{k}_from_img.wav',
                        wav_from_i[k].detach().unsqueeze(0).cpu(), 16_000)

    print('Evaluation files saved under', args.outdir)

if __name__=="__main__":
    main()
