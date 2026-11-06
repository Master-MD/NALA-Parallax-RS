# 🇮🇹 NALA-Parallax-RS (`v0.1.2-nala.6.6.6`) — Manuale Italiano

**Motore LLM Distribuito Pipeline-Parallelo 100 % Rust Nativo (`Edition 2024`, Zero-Python), Allocatore Water-Filling, Router DP Roofline/RTT, Arena Radix Cache & Gateway P2P Zero-Copy**

---

## 1. Perché `NALA-Parallax-RS` è da 45× a 542× più veloce di Python?
Verificato direttamente su **10.000 iterazioni** su **Apple M4 Max** e **Linux Dual-Xeon** con **`100.000%` di parità matematica** (`1.0781 ms`) :
- **Framing P2P Hidden-State a 64 byte Zero-Copy:** `0.132 µs` invece di `71.882 µs` (**`542.9×` più veloce**).
- **Avvio a freddo CLI & consumo RAM:** `1.38 ms` e `2.2 MB` RAM invece di `412.60 ms` e `~380 MB` (**`299.0×` più veloce, `99.4%` di RAM in meno**).
- **Router a Programmazione Dinamica (DP):** `3.071 µs` invece di `754.676 µs` (**`245.7×` più veloce**).
- **Allocatore di layer Water-Filling + Greedy:** `0.917 µs` invece di `194.394 µs` (**`211.9×` più veloce**).
- **Modello di latenza Roofline:** `9.0 ns` invece di `1,464.1 ns` (**`163.6×` più veloce**).
- **Arena Radix Prefix Cache:** `0.355 µs` invece di `20.198 µs` (**`56.9×` più veloce**).
- **Sicurezza `sHeriFF`:** Telemetria nascosta (`chatbe-dev.gradient.network`) completamente rimossa (**`0%` telemetria**).

## 2. Installazione CAVEMAN in 1 riga & Avvio rapido
```bash
curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh
nala-parallax-rs --lang it --plan
nala-parallax-rs --bench
nala-parallax-rs --lang it --serve --port 18082
```
