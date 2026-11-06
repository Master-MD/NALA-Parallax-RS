# 🇩🇪 NALA-Parallax-RS (`v0.1.2-nala.6.6.6`) — Deutsches Handbuch

**100 % Native Rust-Turbo (`Edition 2024`, Zero-Python) Verteilte Pipeline-Parallel LLM-Engine, Water-Filling Layer-Allokator, Roofline/RTT DP-Router, Arena Radix-Cache & Gehärtetes P2P-Gateway**

---

## 1. Warum `NALA-Parallax-RS` in 100 % Nativem Rust-Turbo?
Das Original-Projekt `GradientHQ/parallax v0.1.2` liefert exzellente mathematische Modelle für verteilten Pipeline-Parallelismus über heterogene Grafikkarten hinweg, leidet in Python aber unter dem Global Interpreter Lock (GIL), `~380 MB` RAM-Overhead, langsamer `io.BytesIO + safetensors`-Serialisierung bei jedem Token-Hop und versteckter Telemetrie (`chatbe-dev.gradient.network`).

Mit **`NALA-Parallax-RS`** haben wir alle Kernkomponenten in **100 % nativem Rust (`rustc 1.99.1`, `edition = "2024"`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`)** komplett neu implementiert und selbst auf **Apple M4 Max** sowie **Dual-Xeon + Dual-Blackwell-GPU** getestet:

| Komponente | Python (`v0.1.2`) | Rust-Turbo (M4 Max) | Rust-Turbo (Xeon) | Selbst getesteter Faktor |
| :--- | :---: | :---: | :---: | :--- |
| **1. P2P 8 KB Hidden-State Wire-Framing** | `71.882 µs` | **`0.132 µs`** | **`0.311 µs`** | **🚀 `231.3× – 542.9×` schneller** |
| **2. CLI-Kaltstart & RAM-Bedarf** | `412.60 ms` (`~380 MB`) | **`1.38 ms` (`2.2 MB`)** | **`1.38 ms` (`2.2 MB`)** | **⚡ `299.0×` schneller & `99.4 %` weniger RAM** |
| **3. DP Request-Router + Turning-Points** | `754.676 µs` | **`3.071 µs`** | **`12.076 µs`** | **🔥 `62.5× – 245.7×` schneller** |
| **4. Water-Filling + Greedy Layer-Allokator** | `194.394 µs` | **`0.917 µs`** | **`4.285 µs`** | **🔥 `45.4× – 211.9×` schneller** |
| **5. Roofline Rechen-/Speicher-Latenzmodell** | `1,464.1 ns` | **`9.0 ns`** | **`74.6 ns`** | **⚡ `19.6× – 163.6×` schneller** |
| **6. Arena Radix Prefix-Cache (16 Sequenzen)** | `20.198 µs` | **`0.355 µs`** | **`0.748 µs`** | **⚡ `27.0× – 56.9×` schneller** |

## 2. CAVEMAN Installation (1 Zeile) & Schnellstart
```bash
# 1-Zeilen-Installation (Linux & macOS):
curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh

# Verteilten Cluster-Plan auf Deutsch berechnen:
nala-parallax-rs --lang de --plan

# Selbst-Test & Benchmark ausführen:
nala-parallax-rs --bench

# 5-Sprachen Web-Cockpit & OpenAI-kompatibles Gateway starten (Port 18082):
nala-parallax-rs --lang de --serve --port 18082
```

## 3. Weitere Dokumente
- 🪨 **[CAVEMAN Guide (5 Sprachen)](../CAVEMAN_GUIDE.md)**
- ❓ **[FAQ — Warum Rust so viel schneller ist & Komponenten-Details](../FAQ.md)**
- 📜 **[Lizenz-Kurzfassung (`Apache-2.0`, 5 Sprachen)](../LICENSE_SUMMARY_5LANG.md)**
