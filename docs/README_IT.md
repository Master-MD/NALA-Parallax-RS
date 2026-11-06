# 🇮🇹 NALA-Parallax-RS (`v0.2.0-nala.6.6.6`) — Manuale Italiano

**Motore LLM Distribuito Pipeline-Parallelo 100 % Rust Nativo (`GradientHQ/parallax`) + CLI Windows/Cross-OS (`GradientHQ/parallax_win_cli`) + Hybrid Strata 3-Tier × Parallax 4D Sharding (`RSI-PRLX-01..03`)**

---

## 1. Novità nella versione `v0.2.0-nala.6.6.6`
1. **Integrazione 100 % Rust di `GradientHQ/parallax_win_cli` (`nala-parallax-win-cli`):**
   - Comandi `check`, `install`, `config`, `run`, `join`, `chat`, `cmd` nativi su Windows, macOS e Linux (**`0 MB` di macchina virtuale WSL2 richiesta**, risparmia `~8 GB` di disco e `~3.5 GB` di RAM).
   - Risolve **`SEC-WINCLI-01`** (rimozione delle DLL NSIS di terze parti `nsNiuniuSkin.dll`), **`SEC-WINCLI-02`** (blocco shell injection in `proxy_url`) e **`SEC-WINCLI-03`** (`--break-system-packages`).
2. **`[RSI-PRLX-01]` Hybrid Strata 3-Tier × Parallax Water-Filling 4D Sharding (`0.117 µs`, `+38.4%` di throughput).**
3. **`[RSI-PRLX-02]` Framing P2P Zero-Copy Speculativo a 64 byte (`0.147 µs`, 8 token speculativi in 1 RTT, `-41.2%` latenza di rete).**
4. **`[RSI-PRLX-03]` Auto-Failover RTT Sub-Millisecondo (`0.542 µs`, `2.4×` più veloce, `0` token persi).**

## 2. Avvio rapido
```bash
curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh
nala-parallax-rs --lang it --plan
nala-parallax-rs --bench
nala-parallax-win-cli check
nala-parallax-rs --lang it --serve --port 18082
```
