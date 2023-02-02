#!/bin/bash
wav_dir=../data/magnatagatune/wav/

if [ ! -d "${wav_dir}" ]; then
    cp -r ../data/magnatagatune/mp3/ ${wav_dir}
else
  echo "Directory 'wav' already exists"
fi

for dir in ${wav_dir}/*/; do
    for file in "$dir"*.mp3; do
        filename="${file%.*}"
        ffmpeg -i "$file" "${filename}.wav"
        rm -f $file
    done
done
