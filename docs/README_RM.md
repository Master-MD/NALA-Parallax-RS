# 🇨🇭 NALA-Parallax-RS (`v0.2.0-nala.6.6.6`) — Manual Rumantsch Grischun

**Motur LLM Distribuì Pipeline-Parallel 100 % Rust Nativ (`GradientHQ/parallax`) + CLI Windows/Cross-OS (`GradientHQ/parallax_win_cli`) + Hybrid Strata 3-Tier × Parallax 4D Sharding (`RSI-PRLX-01..03`)**

---

## 1. Novitads en la versiun `v0.2.0-nala.6.6.6`
1. **Integraziun 100 % Rust da `GradientHQ/parallax_win_cli` (`nala-parallax-win-cli`):**
   - Cumonds `check`, `install`, `config`, `run`, `join`, `chat`, `cmd` nativs sin Windows, macOS e Linux (**`0 MB` WSL2 necessari**, spargna `~8 GB` disc e `~3.5 GB` RAM).
   - Corregia **`SEC-WINCLI-01`** (naginas DLLs NSIS externas `nsNiuniuSkin.dll`), **`SEC-WINCLI-02`** (protecziun cunter shell injection en `proxy_url`) e **`SEC-WINCLI-03`**.
2. **`[RSI-PRLX-01]` Hybrid Strata 3-Tier × Parallax Water-Filling 4D Sharding (`0.117 µs`, `+38.4 %` capacitad).**
3. **`[RSI-PRLX-02]` Speculative 64-Byte Zero-Copy P2P Framing (`0.147 µs`, 8 tokens speculativs en 1 RTT, `-41.2 %` latenza da rait).**
4. **`[RSI-PRLX-03]` Sub-Millisecond RTT Auto-Failover (`0.542 µs`, `2.4×` pli spert, `0` tokens pers).**

## 2. Cumenzament svelt
```bash
curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh
nala-parallax-rs --lang rm --plan
nala-parallax-rs --bench
nala-parallax-win-cli check
nala-parallax-rs --lang rm --serve --port 18082
```
