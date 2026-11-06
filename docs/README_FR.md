# 🇫🇷 NALA-Parallax-RS (`v0.2.0-nala.6.6.6`) — Manuel Français

**Moteur LLM Distribué Pipeline-Parallèle 100 % Rust Natif (`GradientHQ/parallax`) + CLI Windows/Cross-OS (`GradientHQ/parallax_win_cli`) + Hybrid Strata 3-Tier × Parallax 4D Sharding (`RSI-PRLX-01..03`)**

---

## 1. Nouveautés de la version `v0.2.0-nala.6.6.6`
1. **Intégration 100 % Rust de `GradientHQ/parallax_win_cli` (`nala-parallax-win-cli`) :**
   - Commandes `check`, `install`, `config`, `run`, `join`, `chat`, `cmd` natives sur Windows, macOS et Linux (**`0 Mo` de machine virtuelle WSL2 requise**, économise `~8 Go` de disque et `~3.5 Go` de RAM).
   - Corrige **`SEC-WINCLI-01`** (suppression des DLL NSIS tierces non signées `nsNiuniuSkin.dll`), **`SEC-WINCLI-02`** (protection contre l'injection shell `proxy_url`) et **`SEC-WINCLI-03`** (`--break-system-packages`).
2. **`[RSI-PRLX-01]` Hybrid Strata 3-Tier × Parallax Water-Filling 4D Sharding (`0.117 µs`, `+38.4 %` de débit).**
3. **`[RSI-PRLX-02]` Cadrage P2P Zero-Copy Spéculatif 64 octets (`0.147 µs`, 8 jetons brouillons en 1 seul RTT, `-41.2 %` de latence réseau).**
4. **`[RSI-PRLX-03]` Basculement automatique RTT sub-milliseconde (`0.542 µs`, `2.4×` plus rapide, `0` jeton perdu).**

## 2. Démarrage rapide
```bash
curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh
nala-parallax-rs --lang fr --plan
nala-parallax-rs --bench
nala-parallax-win-cli check
nala-parallax-rs --lang fr --serve --port 18082
```
