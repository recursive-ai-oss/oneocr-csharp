#!/usr/bin/env bash
# Installs the coverage-guided fuzzing toolchain under one directory, nothing system-wide, no sudo:
#   $FUZZ_TOOLS/llvm/                 clang 20.1.8 release build (only for its libFuzzer runtime)
#   $FUZZ_TOOLS/libfuzzer-dotnet      the SharpFuzz host, compiled here with -fsanitize=fuzzer
#   $FUZZ_TOOLS/sharpfuzz/sharpfuzz   the SharpFuzz instrumenter (dotnet tool)
# Needs: bash, curl, tar, xz, sha256sum, and a .NET SDK on PATH (DOTNET_ROOT set if it is not the default).
# Re-running skips what is already there. Downloads are checked against the SHA-256 values below.
set -euo pipefail

FUZZ_TOOLS="${FUZZ_TOOLS:-$HOME/oneocr-fuzz-tools}"
LLVM_VERSION=20.1.8
LLVM_SHA256=1ead36b3dfcb774b57be530df42bec70ab2d239fbce9889447c7a29a4ddc1ae6
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/LLVM-$LLVM_VERSION-Linux-X64.tar.xz"
HOST_SHA256=90f019e2e9ad3a0b93c7ecc2c5afb2fbfc8b5aab6aac51c7e0d349ec79354f36
HOST_URL="https://raw.githubusercontent.com/Metalnem/libfuzzer-dotnet/master/libfuzzer-dotnet.cc"
SHARPFUZZ_CLI_VERSION="${SHARPFUZZ_CLI_VERSION:-2.2.0}"

mkdir -p "$FUZZ_TOOLS/dl"
cd "$FUZZ_TOOLS"

fetch() { # url file sha256
  if [ ! -f "dl/$2" ]; then curl -fsSL -o "dl/$2.part" "$1" && mv "dl/$2.part" "dl/$2"; fi
  echo "$3  dl/$2" | sha256sum -c --quiet -
}

if [ ! -x llvm/bin/clang++ ]; then
  fetch "$LLVM_URL" "llvm.tar.xz" "$LLVM_SHA256"
  mkdir -p llvm && tar -xJf dl/llvm.tar.xz -C llvm --strip-components=1
fi

if [ ! -f libfuzzer-dotnet ]; then
  fetch "$HOST_URL" "libfuzzer-dotnet.cc" "$HOST_SHA256"
  llvm/bin/clang++ -O2 -fsanitize=fuzzer -o libfuzzer-dotnet dl/libfuzzer-dotnet.cc
fi

if [ ! -x sharpfuzz/sharpfuzz ]; then
  DOTNET_ROLL_FORWARD=Major dotnet tool install SharpFuzz.CommandLine --version "$SHARPFUZZ_CLI_VERSION" --tool-path sharpfuzz
fi

echo "fuzz tools ready in $FUZZ_TOOLS"
