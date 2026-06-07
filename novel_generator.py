import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

# Citations:
#   Mattei, P.-A. & Frellsen, J. (2019). MIWAE: Deep Generative Modelling and
#   Imputation of Incomplete Data via Importance Weighted Autoencoders. AISTATS.
#   Goodfellow, I. et al. (2014). Generative Adversarial Nets. NeurIPS.
#   Mescheder, L. et al. (2017). Adversarial Variational Bayes: Unifying Variational
#   Autoencoders and Generative Adversarial Networks. ICML.

""" Methodology: We extend MIWAE (Mattei & Frellsen, 2019) with a downstream classification head and an adversarial critic 
(Goodfellow et al., 2014; Mescheder et al., 2017), jointly optimizing an IWAE bound, task loss, and GAN loss. 
This task-aware adversarial VAE yields sharper imputations that directly boost predictive performance under arbitrary missingness."""


""" 
Methodology:
 We extend MIWAE (Mattei & Frellsen 2019) with:
   (1) a downstream classification head on the latent mean,
   (2) an adversarial critic trained with spectral‐normed hinge‐loss 
       (Goodfellow 2014; Mescheder 2017) — sharpened reconstructions,
   (3) n_critic discriminator steps per generator update.

Why it can outperform vanilla MIWAE:
 - **Task‐aware supervision**: CE on latent mean steers codes to be discriminative.
 - **Adversarial sharpening**: hinge‐loss critic + spectral‐norm prevent overly‐smooth imputations.
 - **Multi‐step critic**: more robust GAN training under heavy missingness.
"""
class AdversarialTaskVAEGenerator(nn.Module):
    """
    Task-aware, adversarial MIWAE for missing-data imputation.
    Learns via three losses:
      1) IWAE bound (MIWAE)             
      2) classification head supervision
      3) adversarial GAN loss (critic)
    """
    def __init__(self,
                 input_dim: int,
                 n_classes: int,
                 latent_dim: int = 64,
                 hidden_dims=(256, 256),
                 k_iw: int = 20,
                 device='cpu',
                 lambda_cls: float = 1.0,
                 lambda_adv: float = 1e-3,
                 n_critic: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.k_iw = k_iw
        self.device = device
        self.lambda_cls = lambda_cls
        # Feature standardization (fitted during .fit())
        self._feat_mean = None
        self._feat_std = None
        # Feature-importance weights for reconstruction loss (set via set_feature_weights)
        self._feat_weights = None
        self.lambda_adv = lambda_adv
        self.n_critic = n_critic 

        # Encoder q(z|x_obs,mask)
        enc_layers = []
        in_dim = 2 * input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder p(x_miss | z, x_obs, mask) -- heteroscedastic (learned per-feature variance)
        dec_layers = []
        dec_in = latent_dim + input_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(dec_in, h), nn.ReLU()]
            dec_in = h
        self.decoder_backbone = nn.Sequential(*dec_layers)
        self.dec_mu = nn.Linear(dec_in, input_dim)       # mean
        self.dec_logvar = nn.Linear(dec_in, input_dim)    # log-variance
        # Initialize log-variance bias to 0 (unit variance initially)
        nn.init.zeros_(self.dec_logvar.weight)
        nn.init.zeros_(self.dec_logvar.bias)

        # Classification head on latent mean
        self.cls_head = nn.Linear(latent_dim, n_classes)

        # Discriminator: real vs. imputed
        disc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            disc_layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2)]
            in_dim = h
        disc_layers += [nn.Linear(in_dim, 1)]
        
        sn_layers = []
        for layer in disc_layers:
            if isinstance(layer, nn.Linear):
                # after spectral-norm, init bias of final layer to +1
                sn = spectral_norm(layer)
                if layer is disc_layers[-1]:
                    nn.init.constant_(sn.bias, 1.0)
                sn_layers.append(sn)
            else:
                sn_layers.append(layer)
        self.discriminator = nn.Sequential(*sn_layers)

        self.to(device)

    def set_feature_weights(self, classifier):
        """
        Extract feature importance from a trained classifier and use it
        to weight the reconstruction loss. Features that appear in high-level
        tree splits get higher reconstruction priority.

        This makes the generator "task-aware" at the feature level:
        the IWAE bound is weighted so that accurately imputing decision-relevant
        features matters more than imputing irrelevant ones.

        Args:
            classifier: trained sklearn classifier with feature_importances_ attribute
                        (DecisionTreeClassifier, RandomForestClassifier, etc.)
        """
        if hasattr(classifier, 'feature_importances_'):
            imp = classifier.feature_importances_
            # Normalize to mean=1 so total loss magnitude doesn't change
            imp = imp / (imp.mean() + 1e-8)
            # Smooth: don't let any feature have zero weight
            imp = np.clip(imp, 0.1, None)
            imp = imp / imp.mean()  # re-normalize after clipping
            self._feat_weights = torch.from_numpy(imp.astype(np.float32)).to(self.device)
        else:
            self._feat_weights = None

    def forward(self, x, mask):
        B, d = x.shape
        inp = torch.cat([x * mask, mask], dim=1)
        h = self.encoder(inp)
        mu = self.fc_mu(h)
        
        logvar = self.fc_logvar(h).clamp(-10, 10)
        std = (0.5 * logvar).exp().clamp_min(1e-6)

        # MIWAE: draw k samples
        mu_k = mu.unsqueeze(1).expand(B, self.k_iw, self.latent_dim)
        std_k = std.unsqueeze(1).expand(B, self.k_iw, self.latent_dim)
        eps = torch.randn_like(std_k)
        z = mu_k + std_k * eps  # (B, K, z)

        # Decode (heteroscedastic: outputs mean + log-variance)
        z_flat = z.reshape(B * self.k_iw, self.latent_dim)
        mask_rep = mask.unsqueeze(1).expand(B, self.k_iw, d).reshape(B*self.k_iw, d)
        dec_inp = torch.cat([z_flat, mask_rep], dim=1)
        h_dec = self.decoder_backbone(dec_inp)
        recon_mu = self.dec_mu(h_dec).reshape(B, self.k_iw, d)
        recon_logvar = self.dec_logvar(h_dec).clamp(-6, 2).reshape(B, self.k_iw, d)
        return recon_mu, recon_logvar, mu, logvar, z, std_k

    def fit(self, X: np.ndarray, y: np.ndarray,
            num_epochs: int = 100,
            batch_size: int = 128,
            lr: float = 1e-3,
            adv_warmup_epochs: int = 20,
            ):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Feature standardization: train in normalized space
        self._feat_mean = X.mean(axis=0).astype(np.float32)
        self._feat_std = X.std(axis=0).astype(np.float32)
        self._feat_std[self._feat_std < 1e-8] = 1.0  # avoid div-by-zero
        X_normed = (X - self._feat_mean) / self._feat_std

        ds = TensorDataset(torch.from_numpy(X_normed).float(), torch.from_numpy(y).long())

        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # ────────────────────────────────────────────
        # only update {encoder, decoder, cls_head} in the generator step
        gen_params = []
        gen_params += list(self.encoder.parameters())
        gen_params += list(self.fc_mu.parameters())
        gen_params += list(self.fc_logvar.parameters())
        gen_params += list(self.decoder_backbone.parameters())
        gen_params += list(self.dec_mu.parameters())
        gen_params += list(self.dec_logvar.parameters())
        gen_params += list(self.cls_head.parameters())
        optim_g    = optim.Adam(gen_params, lr=lr)

        # only update the discriminator in the critic step
        disc_params = list(self.discriminator.parameters())
        optim_d     = optim.Adam(disc_params, lr=lr*2, weight_decay=1e-4)
        # ────────────────────────────────────────────

        # use logits-loss (combines a sigmoid+BCELoss under the hood)
        def d_hinge_loss(real_scores, fake_scores):
            # real loss: max(0, 1 - D(real))
            # fake loss: max(0, 1  D(fake))
            loss_real = torch.mean(F.relu(1.0 - real_scores))
            loss_fake = torch.mean(F.relu(1.0 + fake_scores))
            return loss_real + loss_fake
        
        celoss = nn.CrossEntropyLoss()

        for epoch in range(1, num_epochs+1):
            # Adversarial warmup scheduling
            if epoch <= adv_warmup_epochs:
                current_lambda_adv = 0.0
            else:
                warmup_progress = min(1.0, (epoch - adv_warmup_epochs) / adv_warmup_epochs)
                current_lambda_adv = self.lambda_adv * warmup_progress

            total_g, total_d = 0.0, 0.0
            for batch_idx, (batch, labels) in enumerate(dl, start=1):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                B, d = batch.shape
                p = torch.rand(B, 1, device=self.device)*0.6 + 0.2
                mask = (torch.rand_like(batch) > p).float()

                # === Generator forward ===
                recon_mu, recon_logvar, mu, logvar, z, std_k = self(batch, mask)
                # MIWAE loss with heteroscedastic decoder
                x_rep = batch.unsqueeze(1).expand(B, self.k_iw, d)
                mask_e = mask.unsqueeze(1).expand(B, self.k_iw, d)
                # Heteroscedastic Gaussian log-likelihood: -0.5*(sq_err/var + log_var)
                sq_err = (recon_mu - x_rep)**2 * (1-mask_e)  # (B, K, d)
                recon_var = recon_logvar.exp()  # (B, K, d)
                if self._feat_weights is not None:
                    sq_err = sq_err * self._feat_weights.unsqueeze(0).unsqueeze(0)
                # log p(x_miss|z) = sum_j [-0.5*(x_j - mu_j)^2/var_j - 0.5*log(var_j)]
                log_px = -0.5 * ((sq_err / (recon_var + 1e-8)) + recon_logvar * (1-mask_e)).sum(dim=2)
                log_pz = -0.5*(z**2).sum(dim=2)
                log_qz = -0.5*(((z - mu.unsqueeze(1))**2/(std_k**2)) + logvar.unsqueeze(1)).sum(dim=2)
                log_w = log_px + log_pz - log_qz
                iw = torch.logsumexp(log_w, dim=1) - math.log(self.k_iw)
                loss_iw = -iw.mean()

                # Classification loss on latent mean
                cls_logits = self.cls_head(mu)
                loss_cls = celoss(cls_logits, labels)

                # Adversarial loss (generator fools discriminator)
                if current_lambda_adv > 0 and self.n_critic > 0:
                    fake = recon_mu.reshape(B*self.k_iw, d)
                    adv_loss = -self.discriminator(fake).mean()
                else:
                    adv_loss = torch.tensor(0.0, device=self.device)

                # Total generator loss (with warmup-scheduled adversarial weight)
                loss_g = loss_iw + self.lambda_cls*loss_cls + current_lambda_adv*adv_loss
                optim_g.zero_grad(); 
                loss_g.backward(); 
                optim_g.step()
                total_g += loss_g.item()

                # ——— Spectral-Norm Hinge Discriminator update ———
                for _ in range(self.n_critic if self.lambda_adv > 0 else 0):
                    real_scores = self.discriminator(batch)

                    with torch.no_grad():
                        recon_mu2, recon_logvar2, *_ = self(batch, mask)
                        fake2 = recon_mu2.reshape(B*self.k_iw, d)

                    fake_scores = self.discriminator(fake2.detach())
                    loss_d = d_hinge_loss(real_scores, fake_scores)

                    optim_d.zero_grad()
                    loss_d.backward()

                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    optim_d.step()
                    total_d += loss_d.item()

                # debug only on first epoch & first batch
                # if epoch == 1 and batch_idx == 1:
                #     print(f"[DBG][D] real_scores → "
                #           f"μ={real_scores.mean():.4f}  σ={real_scores.std():.4f}")
                #     print(f"[DBG][D] fake_scores → "
                #           f"μ={fake_scores.mean():.4f}  σ={fake_scores.std():.4f}")
                #     print(f"[DBG][D] hinge_loss_d = {loss_d.item():.4f} "
                #           f"(expected ≈5–10)")

                
                # if batch_idx % 2 == 0:
                #     print(f"[LOSS] IWAE={loss_iw.item():.4f}   CLS={loss_cls.item():.4f}   ADV={adv_loss.item():.4f}")


            if epoch % 10 == 0:
                avg_d = total_d / len(dl)
                print(f"[ATVAE] Epoch {epoch}/{num_epochs}"
                      f"  G_loss={total_g/len(dl):.4f}"
                      f"  D_loss_per_batch≈{avg_d:.4f}")


    def sample_latent(self, x_obs: np.ndarray, mask: np.ndarray, rng: np.random.RandomState):
        """Return the posterior mean and std for the latent code (used by Rao-Blackwell)."""
        self.eval()
        x_clean = np.nan_to_num(x_obs, nan=0.0)
        if self._feat_mean is not None:
            x_normed = (x_clean - self._feat_mean) / self._feat_std
        else:
            x_normed = x_clean
        with torch.no_grad():
            x = torch.from_numpy(x_normed).float().unsqueeze(0).to(self.device)
            m = torch.from_numpy(mask.astype(float)).float().unsqueeze(0).to(self.device)
            inp = torch.cat([x * m, m], dim=1)
            h = self.encoder(inp)
            mu = self.fc_mu(h).squeeze(0).cpu().numpy()
            logvar = self.fc_logvar(h).clamp(-10, 10).squeeze(0).cpu().numpy()
            std = np.exp(0.5 * logvar)
        return mu, std

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': [l.out_features for l in self.encoder if isinstance(l, nn.Linear)],
            'n_classes': self.n_classes,
            'k_iw': self.k_iw,
            'lambda_cls': self.lambda_cls,
            'lambda_adv': self.lambda_adv,
            'n_critic': self.n_critic,
            'feat_mean': self._feat_mean,
            'feat_std': self._feat_std,
        }, path)

    @classmethod
    def load(cls, path: str, device='cpu'):
        data = torch.load(path, map_location=device)
        gen = cls(input_dim=data['input_dim'],
                  n_classes=data['n_classes'],
                  latent_dim=data['latent_dim'],
                  hidden_dims=data['hidden_dims'],
                  k_iw=data['k_iw'],
                  device=device,
                  lambda_cls=data['lambda_cls'],
                  lambda_adv=data['lambda_adv'],
                  n_critic=data['n_critic'])
        gen.load_state_dict(data['state_dict'])
        gen._feat_mean = data.get('feat_mean', None)
        gen._feat_std = data.get('feat_std', None)
        gen.eval()
        return gen

    def sample(self, x_obs: np.ndarray, mask: np.ndarray, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        self.eval()
        d = self.input_dim

        # Replace NaN with 0 before processing (NaN * 0 = NaN in IEEE 754)
        x_clean = np.nan_to_num(x_obs, nan=0.0)

        # Standardize input (generator operates in normalized space)
        if self._feat_mean is not None:
            x_normed = (x_clean - self._feat_mean) / self._feat_std
        else:
            x_normed = x_clean

        reps = int(np.ceil(n_samples / self.k_iw))
        out = []
        for _ in range(reps):
            with torch.no_grad():
                x = torch.from_numpy(x_normed).float().unsqueeze(0).to(self.device)
                m = torch.from_numpy(mask.astype(float)).float().unsqueeze(0).to(self.device)
                recon_mu, recon_logvar, _, _, _, _ = self(x, m)
                # Sample from heteroscedastic decoder: x ~ N(mu, exp(logvar))
                recon_std = (0.5 * recon_logvar).exp()
                eps = torch.randn_like(recon_mu)
                samples = recon_mu + eps * recon_std
                arr = samples.squeeze(0).cpu().numpy()
            out.append(arr)
        S = np.vstack(out)[:n_samples]

        # Unstandardize output back to original space
        if self._feat_mean is not None:
            S = S * self._feat_std + self._feat_mean

        S[:, mask==1] = x_clean[mask==1]
        return S
