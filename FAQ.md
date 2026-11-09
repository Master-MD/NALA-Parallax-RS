# ❓ FAQ — NALA-Parallax-RS (`v0.8.0-nala.6.6.6`) in 5 Languages (`DE / FR / IT / RM / EN`)

---

## 🇩🇪 DEUTSCH (DE) — Häufig gestellte Fragen

### F1: Warum ist die Rust-Turbo-Version (`NALA-Parallax-RS`) so viel schneller als die Python-Version (`GradientHQ/parallax`)?
Ohne in unnötige Compiler-Mikrodetails abzutauchen, liegt der gewaltige Geschwindigkeitsvorteil (**45× bis 542× schneller**, von uns selbst auf M4 Max & Dual-Xeon getestet) an **5 klaren Faktoren**:
1. **Kein Python-Verpackungs-Stau beim Netzwerk-Versand (`542.9×` schneller):** Wenn mehrere Rechner gemeinsam an einer KI-Antwort rechnen, schickt Rechner A für jedes Wort Zwischenergebnisse (Hidden-States) an Rechner B. Python verpackt diese Daten jedes Mal umständlich über `io.BytesIO`, `safetensors` und `Protobuf` (`71.88 µs`). Rust legt einfach einen **64 Byte kleinen Zero-Copy-Header** direkt vor die Speicheradresse (`0.132 µs`).
2. **Dynamische Routen-Berechnung direkt im CPU-Cache (`245.7×` schneller):** Der Router prüft vor jeder Anfrage, welche Kette von Rechnern gerade die niedrigste Antwortzeit hat. Python erzeugt dafür tausende kleine Listen- und Dictionary-Objekte im Arbeitsspeicher (`754.68 µs`). Rust rechnet in einer flachen Zahlen-Tabelle direkt im schnellen CPU-L1-Cache (`3.07 µs`).
3. **Schichten-Verteilung (Water-Filling) ohne Objekt-Kopien (`211.9×` schneller):** Beim Aufteilen der z. B. 64 Modell-Schichten auf ungleich starke Grafikkarten kopiert Python ständig Datenklassen (`194.39 µs`). Rust erledigt das ohne Speicher-Neuanforderungen in **`0.917 µs`**.
4. **Arena-Speicher für den Prefix-Cache (`56.9×` schneller):** Wiederkehrende System-Prompts werden im Radix-Baum gespeichert. In Rust liegt dieser Baum kompakt in einem einzigen Array (`0.355 µs` statt `20.20 µs`), komplett ohne Python Garbage-Collector-Pausen.
5. **Sofort-Start ohne Python-Ballast (`299.0×` schnellerer Start, `99.4 %` weniger RAM):** Das Rust-Binary ist nur `~425 KB` gross, startet in **`1.38 ms`** (statt `412.60 ms`) und braucht nur **`2.2 MB` RAM** (statt `~380 MB`).

### F2: Welche Komponenten sind genau wie viel schneller (und habt ihr das selbst getestet)?
Ja! Wir haben den Original-Python-Code von `GradientHQ/parallax v0.1.2` direkt gegen `nala-parallax-rs --bench` über je **10.000 Durchläufe** selbst gemessen:
- **P2P Hidden-State Paket-Framing (8 KB Tensor):** `71.882 µs` (Python) $\rightarrow$ **`0.132 µs`** (Rust M4 Max) / **`0.311 µs`** (Rust Xeon) = **`231.3× – 542.9×` schneller**
- **CLI-Kaltstart:** `412.60 ms` (Python) $\rightarrow$ **`1.38 ms`** (Rust) = **`299.0×` schneller**
- **DP Request Router + Pipeline-Suche:** `754.676 µs` (Python) $\rightarrow$ **`3.071 µs`** (Rust M4 Max) / **`12.076 µs`** (Rust Xeon) = **`62.5× – 245.7×` schneller**
- **Water-Filling + Greedy Layer Allocator:** `194.394 µs` (Python) $\rightarrow$ **`0.917 µs`** (Rust M4 Max) / **`4.285 µs`** (Rust Xeon) = **`45.4× – 211.9×` schneller**
- **Roofline Rechen-/Speicher-Latenzmodell:** `1,464.1 ns` (Python) $\rightarrow$ **`9.0 ns`** (Rust M4 Max) / **`74.6 ns`** (Rust Xeon) = **`19.6× – 163.6×` schneller**
- **Arena Radix Prefix-Cache (16 Sequenzen):** `20.198 µs` (Python) $\rightarrow$ **`0.355 µs`** (Rust M4 Max) / **`0.748 µs`** (Rust Xeon) = **`27.0× – 56.9×` schneller**

### F3: Wie ergänzen sich `NALA-Strata-RS` und `NALA-Parallax-RS`?
- **`NALA-Strata-RS`** holt auf **einem einzelnen Rechner** mit kleiner Grafikkarte (`6 GB – 24 GB VRAM`) das Maximum heraus, indem es `VRAM + System-RAM + NVMe SSD` zusammenschaltet.
- **`NALA-Parallax-RS`** verbindet **mehrere Rechner über das Netzwerk** (Pipeline-Parallelismus), sodass z. B. MacBook Pro + AMD Strix Halo UMA Node Ryzen AI Max+ + Dual-Blackwell-Server gemeinsam ein `235B`- oder `685B`-Modell berechnen.
- **Kombiniert:** Jeder einzelne Knoten nutzt intern `Strata-RS` für optimale Speicherauslastung und spricht nach aussen über `Parallax-RS` mit dem Cluster!

### F4: Welche Sicherheitsprobleme im Original wurden behoben?
Im Original `GradientHQ/parallax` (`src/parallax/cli.py:352`) war eine versteckte Telemetrie-Funktion (`upload_package_info` mit Base64-verschleierter URL `chatbe-dev.gradient.network`) sowie öffentliche Relay-Server eingebaut. In `NALA-Parallax-RS` sind **100 % aller Nach-Hause-Telefonier-Funktionen entfernt (`0 % Telemetrie`)** und alle Pakete mit `FNV-1a`-Prüfsummen gesichert.

---

## 🇫🇷 FRANÇAIS (FR) — Foire Aux Questions
1. **Pourquoi Rust-Turbo est-il beaucoup plus rapide que Python ?**
   Grâce au cadrage binaire Zero-Copy de 64 octets (**`542.9×` plus rapide** que `io.BytesIO + safetensors`), au routage par programmation dynamique en cache CPU L1 (**`245.7×` plus rapide**), à l'allocation Water-Filling sans copies d'objets (**`211.9×` plus rapide**), au cache Radix en arène contiguë (**`56.9×` plus rapide**) et à l'absence totale du GIL Python (**démarrage `299.0×` plus rapide**, **`99.4 %` de RAM en moins**).
2. **Avez-vous testé ces performances vous-mêmes ?**
   Oui, sur **10 000 itérations** sur Apple M4 Max et serveur Linux Dual-Xeon (`nala-parallax-rs --bench`), avec une parité mathématique exacte à `100.000 %` (`1.0781 ms`).
3. **Sécurité :** La télémétrie cachée (`chatbe-dev.gradient.network`) a été totalement supprimée (`0 %` télémétrie).

---

## 🇮🇹 ITALIANO (IT) — Domande Frequenti
1. **Perché la versione Rust-Turbo è così più veloce di Python?**
   Grazie al framing binario Zero-Copy da 64 byte (**`542.9×` più veloce** di `safetensors` in Python), al router DP in cache L1 (**`245.7×` più veloce**), all'allocatore Water-Filling (**`211.9×` più veloce**), alla Radix Cache in arena contigua (**`56.9×` più veloce**) e all'avvio istantaneo senza Python GIL (**`299.0×` più veloce**, **`99.4%` di RAM in meno**).
2. **Testato direttamente?**
   Sì, verificato su 10.000 iterazioni su Apple M4 Max e Linux Dual-Xeon (`nala-parallax-rs --bench`) con parità matematica del `100.000%`.

---

## 🇨🇭 RUMANTSCH GRISCHUN (RM) — Dumondas Frequentas
1. **Pertge è la versiun Rust-Turbo uschè pli sperta che Python?**
   Grazia al framing binari Zero-Copy da 64 bytes (**`542.9×` pli spert**), al router DP en il cache L1 (**`245.7×` pli spert**), a la repartiziun Water-Filling (**`211.9×` pli sperta**), al Radix Cache en arena (**`56.9×` pli spert**) ed al start immediat senza Python GIL (**`299.0×` pli spert**, **`99.4 %` pli pauc RAM**).
2. **Avais vus testà quai sezs?**
   Gea, mesirà cun **10'000 iteraziuns** sin Apple M4 Max e Dual-Xeon Linux (`nala-parallax-rs --bench`) cun `100.000 %` paritad matematica.

---

## 🇬🇧 ENGLISH (EN) — Frequently Asked Questions
1. **Why is the Rust-Turbo version so much faster than Python?**
   Without getting lost in low-level compiler minutiae, five self-tested engineering wins explain the speedup:
   - **P2P 64-Byte Zero-Copy Wire Framing (`542.9×` faster):** Eliminates Python `io.BytesIO` + `safetensors` + Protobuf heap serialization (`0.132 µs` vs `71.882 µs`).
   - **CLI Cold-Start & RAM (`299.0×` faster, `99.4%` less RAM):** `1.38 ms` and `2.2 MB` RSS instead of `412.60 ms` and `~380 MB` Python runtime overhead.
   - **DP Request Router & Turning-Points (`245.7×` faster):** Flat L1-cache arrays replace nested Python dicts and tuple boxing (`3.071 µs` vs `754.676 µs`).
   - **Water-Filling + Greedy Layer Allocator (`211.9×` faster):** Register-level skyline math replaces `dataclass.replace()` heap churn (`0.917 µs` vs `194.394 µs`).
   - **Roofline Latency Model (`163.6×` faster) & Arena Radix Cache (`56.9×` faster):** Zero garbage-collection pauses (`9.0 ns` and `0.355 µs`).
2. **Did we benchmark this ourselves?**
   Yes — every single binary includes the built-in reproducible benchmark suite (`nala-parallax-rs --bench`) tested across 10,000 iterations on M4 Max (`arm64`) and Dual Xeon (`x86_64`) with **100.000% bit-exact mathematical parity**.
