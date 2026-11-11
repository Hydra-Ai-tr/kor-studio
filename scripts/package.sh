#!/bin/sh
# Packs the Linux or macOS release from a finished build into build/release:
#   kor-studio-<version>-<os>-<arch>.tar.gz
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#   sh scripts/package.sh [build-dir]
set -eu
cd "$(dirname "$0")/.."
build="${1:-build}"
version=$(sed -n 's/^project(KorStudio VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)
os=$(uname -s | tr 'A-Z' 'a-z')
[ "$os" = darwin ] && os=macos
arch=$(uname -m)
name="kor-studio-$version-$os-$arch"
out="build/release"
dir="$out/$name"

for exe in kor-cli kor-studio; do
    [ -x "$build/bin/$exe" ] || { echo "$build/bin/$exe is missing: build first" >&2; exit 1; }
done
rm -rf "$dir"
mkdir -p "$dir/assets" "$dir/community" "$dir/examples" "$dir/plugins" "$dir/python/kor"
cp "$build/bin/kor-studio" "$build/bin/kor-cli" LICENSE README.md "$dir/"
cp assets/hero.png "$dir/assets/"
cp community/index.json "$dir/community/"
cp plugins/*.py "$dir/plugins/"
cp python/kor/*.py "$dir/python/kor/"
for f in examples/*.txt examples/*.csv examples/*.py; do
    [ -f "$f" ] && cp "$f" "$dir/examples/"
done
tar -czf "$dir.tar.gz" -C "$out" "$name"
echo "wrote $dir.tar.gz"
