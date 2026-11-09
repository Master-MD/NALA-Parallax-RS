#!/bin/sh
# ============================================================================
# 🦀 NALA-Parallax-RS (v0.7.0-nala.6.6.6) — 5-Language One-Line Installer (DE/FR/IT/RM/EN)
# Linux (x86_64 & ARM64 NVIDIA DGX Spark) & macOS (Apple Silicon M1-M5 & Intel)
# ============================================================================
#   curl -fsSL https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download/install.sh -o install.sh && sh install.sh
# ============================================================================
set -eu

FEED="${PARALLAX_FEED:-https://github.com/Master-MD/NALA-Parallax-RS/releases/latest/download}"
FEED="${FEED%/}"
DATA_DIR="${PARALLAX_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/parallax}"
BIN_DIR="${PARALLAX_BIN_DIR:-$HOME/.local/bin}"
LANG_CODE="${NALA_LANG:-${LANG:-de}}"

say() { printf '%s\n' "$*"; }
die() { printf 'nala-parallax-rs installer: %s\n' "$*" >&2; exit 1; }

main() {
  case "$LANG_CODE" in
    fr*) say "🦀 Installation de NALA-Parallax-RS (100% Rust Natif, Zéro-Python, Zéro-Télémétrie)..." ;;
    it*) say "🦀 Installazione di NALA-Parallax-RS (100% Rust Nativo, Zero-Python, Zero-Telemetria)..." ;;
    rm*) say "🦀 Installaziun da NALA-Parallax-RS (100% Rust Nativ, Zero-Python, Zero-Telemetria)..." ;;
    en*) say "🦀 Installing NALA-Parallax-RS (100% Native Rust, Zero-Python, Zero-Telemetry)..." ;;
    *)   say "🦀 Installiere NALA-Parallax-RS (100% Native Rust, Zero-Python, Zero-Telemetrie)..." ;;
  esac

  if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
  elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q -O "$2" "$1"; }
  else
    die "curl or wget is required."
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum "$1" | cut -d' ' -f1; }
  elif command -v shasum >/dev/null 2>&1; then
    sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
  else
    die "sha256sum or shasum is required."
  fi

  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "$OS-$ARCH" in
    Linux-x86_64|Linux-amd64)    PLAT="linux-x86_64" ;;
    Linux-aarch64|Linux-arm64)   PLAT="linux-aarch64" ;;
    Darwin-arm64|Darwin-aarch64) PLAT="macos-arm64" ;;
    Darwin-x86_64)               PLAT="macos-x86_64" ;;
    *) die "Unsupported platform: $OS-$ARCH" ;;
  esac

  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT

  fetch "$FEED/install-manifest.txt" "$TMP_DIR/install-manifest.txt"
  LINE="$(grep "^$PLAT " "$TMP_DIR/install-manifest.txt" | head -n 1 || true)"
  [ -n "$LINE" ] || die "No entry for $PLAT in install-manifest.txt"

  VER="$(echo "$LINE" | awk '{print $2}')"
  ARCHIVE="$(echo "$LINE" | awk '{print $3}')"
  EXPECTED_SHA="$(echo "$LINE" | awk '{print $4}')"

  fetch "$FEED/$ARCHIVE" "$TMP_DIR/$ARCHIVE"
  ACTUAL_SHA="$(sha "$TMP_DIR/$ARCHIVE")"
  [ "$ACTUAL_SHA" = "$EXPECTED_SHA" ] || die "SHA-256 mismatch for $ARCHIVE"

  DEST="$DATA_DIR/terminal/$VER"
  mkdir -p "$DEST" "$BIN_DIR"
  tar -xzf "$TMP_DIR/$ARCHIVE" -C "$DEST"

  ln -sf "$DEST/bin/parallax" "$BIN_DIR/parallax"
  ln -sf "$DEST/bin/nala-parallax-rs" "$BIN_DIR/nala-parallax-rs"

  case "$LANG_CODE" in
    fr*) say "✅ Installé avec succès ! Lancez : nala-parallax-rs --lang fr --plan" ;;
    it*) say "✅ Installato con successo! Avvia con: nala-parallax-rs --lang it --plan" ;;
    rm*) say "✅ Installà cun success! Cumenza cun: nala-parallax-rs --lang rm --plan" ;;
    en*) say "✅ Installed. Run: nala-parallax-rs --lang en --plan" ;;
    *)   say "✅ Erfolgreich installiert! Starte mit: nala-parallax-rs --lang de --plan" ;;
  esac
}

main "$@"
