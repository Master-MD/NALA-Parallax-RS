# 🇨🇭 NALA-Parallax-RS (`v0.1.2-nala.6.6.6`) — Manual Rumantsch Grischun

**Motur LLM Distribuì Pipeline-Parallel 100 % Rust Nativ (`Edition 2024`, Zero-Python), Allocatur Water-Filling, Router DP Roofline/RTT, Arena Radix Cache & Gateway P2P Zero-Copy**

---

## 1. Pertge è `NALA-Parallax-RS` 45× fin 542× pli spert che Python?
Testà da nus sezs cun **10'000 iteraziuns** sin **Apple M4 Max** e **Linux Dual-Xeon** cun **`100.000 %` paritad matematica** (`1.0781 ms`) :
- **P2P Hidden-State Framing da 64 bytes Zero-Copy:** `0.132 µs` enstagl da `71.882 µs` (**`542.9×` pli spert**).
- **Start a fraid CLI & diever da RAM:** `1.38 ms` e `2.2 MB` RAM enstagl da `412.60 ms` e `~380 MB` (**`299.0×` pli spert, `99.4 %` pli pauc RAM**).
- **Router da Programaziun Dinamica (DP):** `3.071 µs` enstagl da `754.676 µs` (**`245.7×` pli spert**).
- **Allocatur da stresas Water-Filling + Greedy:** `0.917 µs` enstagl da `194.394 µs` (**`211.9×` pli spert**).
- **Model da latenza Roofline:** `9.0 ns` enstagl da `1,464.1 ns` (**`163.6×` pli spert**).
- **Arena Radix Prefix Cache:** `0.355 µs` enstagl da `20.198 µs` (**`56.9×` pli spert**).
- **Segirezza `sHeriFF`:** Telemetria zuppada (`chatbe-dev.gradient.network`) eliminada dal tuttafatg (**`0 %` telemetria**).

## 2. Installaziun CAVEMAN en 1 lingia & Cumenzament svelt
```bash
curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh
nala-parallax-rs --lang rm --plan
nala-parallax-rs --bench
nala-parallax-rs --lang rm --serve --port 18082
```
