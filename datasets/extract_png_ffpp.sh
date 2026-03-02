# code/DiffusionFake/datasets/extract_png_ffpp.sh

# #!/usr/bin/env bash
# set -euo pipefail

# ROOT=/scratch/sahil/projects/img_deepfake/datasets/ffpp
# FPS=1

# extract_originals () {
#   local in_dir="$ROOT/original_sequences/youtube/raw/videos"
#   local out_dir="$ROOT/original_sequences/youtube/raw/images"
#   mkdir -p "$out_dir"

#   echo "Extracting originals -> $out_dir"
#   while IFS= read -r mp4; do
#     id="$(basename "$mp4" .mp4)"
#     mkdir -p "$out_dir/$id"
#     echo "  ffmpeg: $id"
#     ffmpeg -nostdin -hide_banner -loglevel error \
#       -i "$mp4" -vf "fps=$FPS" -start_number 0 \
#       "$out_dir/$id/${id}_%04d.png"
#   done < <(find "$in_dir" -maxdepth 1 -type f -name "*.mp4" | sort)
# }

# extract_manipulated () {
#   local method="$1"
#   local in_dir="$ROOT/manipulated_sequences/$method/raw/videos"
#   local out_dir="$ROOT/manipulated_sequences/$method/raw/images"
#   mkdir -p "$out_dir"

#   echo "Extracting $method -> $out_dir"
#   while IFS= read -r mp4; do
#     pair="$(basename "$mp4" .mp4)"   # e.g. 174_964
#     mkdir -p "$out_dir/$pair"
#     echo "  ffmpeg: $pair"
#     ffmpeg -nostdin -hide_banner -loglevel error \
#       -i "$mp4" -vf "fps=$FPS" -start_number 0 \
#       "$out_dir/$pair/${pair}_%04d.png"
#   done < <(find "$in_dir" -maxdepth 1 -type f -name "*.mp4" | sort)
# }

# extract_originals
# extract_manipulated Face2Face
# extract_manipulated Deepfakes
# extract_manipulated FaceSwap
# extract_manipulated NeuralTextures

# echo "Done."
# # BASH



# chmod +x ./datasets/extract_png_ffpp.sh


# === Above is the original code. Below is the completed code. ===
ROOT=/scratch/sahil/projects/img_deepfake/datasets/ffpp
COMP=c23

cd "$ROOT/original_sequences/youtube/$COMP" || exit 1

# make c23/000 -> c23/images/000 for every id folder
for d in images/*; do
  b="$(basename "$d")"
  [ -d "$d" ] || continue
  ln -sfn "images/$b" "$b"
done

for m in Deepfakes Face2Face FaceSwap NeuralTextures; do
  cd "$ROOT/manipulated_sequences/$m/$COMP" || continue
  for d in images/*; do
    b="$(basename "$d")"
    [ -d "$d" ] || continue
    ln -sfn "images/$b" "$b"
  done
done
