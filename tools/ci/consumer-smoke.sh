#!/usr/bin/env bash
# Installs the packed packages into a fresh program outside the repository and runs it, on the JIT and as a NativeAOT binary.
#
#   tools/ci/consumer-smoke.sh <packages directory> <version> <rid>      e.g. linux-x64, win-x64, osx-arm64, linux-arm64
#
# NuGet gets two sources: the packages directory for every RecursiveAI.OneOcr* id, nuget.org for the rest, and its own empty
# package cache so nothing comes from a previous build.
set -euo pipefail
pk=$(cd "${1:?usage: consumer-smoke.sh <packages directory> <version> <rid>}" && pwd)
version=${2:?}
rid=${3:?}
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/consumer-smoke" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$src"/Smoke.csproj "$src"/Program.cs "$src"/hello.png "$work/"
cat > "$work/nuget.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="$pk" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="local"><package pattern="RecursiveAI.OneOcr*" /></packageSource>
    <packageSource key="nuget.org"><package pattern="*" /></packageSource>
  </packageSourceMapping>
</configuration>
EOF
export NUGET_PACKAGES="$work/packages"
cd "$work"
echo "== JIT"
dotnet run -c Release -p:OneOcrVersion="$version"
echo "== NativeAOT ($rid)"
dotnet publish -c Release -r "$rid" -p:OneOcrVersion="$version" -o "$work/aot"
exe="$work/aot/Smoke"
[[ -f $exe.exe ]] && exe="$exe.exe"
"$exe"
