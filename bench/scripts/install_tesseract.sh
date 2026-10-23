#!/usr/bin/env bash
# Install the distro's Tesseract binary packages under $TOOLS (default ~/tools/tesseract) without root,
# and fetch the stock English models from the official tessdata_best / tessdata_fast repositories.
# The binary and models are used as a black box by the benchmark; nothing from them is loaded by our code.
set -euo pipefail

TOOLS="${TOOLS:-$HOME/tools/tesseract}"
# Pinned tessdata commits would be ideal; the files are hashed into versions.txt instead.
BEST_URL="https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"
FAST_URL="https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata"
OSD_BEST_URL="https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata"

mkdir -p "$TOOLS/debs" "$TOOLS/root" "$TOOLS/best" "$TOOLS/fast"
cd "$TOOLS/debs"

# Everything tesseract-ocr needs that is not already installed (simulated, no root).
pkgs=$(apt-get install -s --no-install-recommends tesseract-ocr | awk '/^Inst /{print $2}')
apt-get download $pkgs
for d in *.deb; do dpkg -x "$d" "$TOOLS/root"; done

cd "$TOOLS"
for k in best fast; do
  url_var=${k^^}_URL
  curl -fsSL -o "$k/eng.traineddata" "${!url_var}"
done
curl -fsSL -o best/osd.traineddata "$OSD_BEST_URL"
cp best/osd.traineddata fast/osd.traineddata

cat > "$TOOLS/tess" <<EOF
#!/usr/bin/env bash
# usage: tess best|fast <tesseract args...>
k=\$1; shift
export LD_LIBRARY_PATH="$TOOLS/root/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
exec "$TOOLS/root/usr/bin/tesseract" --tessdata-dir "$TOOLS/\$k" "\$@"
EOF
chmod +x "$TOOLS/tess"

{
  echo "# installed $(date -Is)"
  "$TOOLS/tess" best --version 2>&1 | head -8
  echo "packages: $(echo $pkgs | tr '\n' ' ')"
  sha256sum best/eng.traineddata fast/eng.traineddata best/osd.traineddata
} | tee "$TOOLS/versions.txt"
