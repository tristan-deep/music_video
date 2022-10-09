"""Author Tristan Stevens

Inspired by Keras tutorial: 
    https://keras.io/examples/generative/random_walks_with_stable_diffusion/

pip install whisper
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 
pip install tensorflow-gpu
pip install keras-cv
pip install pillow
"""


import datetime
import glob
import os
from pathlib import Path

import cv2
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import srt
import tensorflow as tf
import tqdm
from PIL import Image
from tensorflow import keras
from pprint import pprint
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

img_width=512 + 128
img_height=512
seed = 123
fps = 30
batch_size = 3
unconditional_guidance_scale = 7.5
style = 'oil painting'

def extract_lyrics(file, save_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    file = str(file)
    os.system(
        f"whisper {file} --model medium  --language English --output_dir {str(save_folder)} --verbose False --device {device}"
    )

    lyrics_file = save_folder / (file + ".srt")
    return lyrics_file

def read_lyrics(file, fps):

    with open(file) as f:
        data = f.read()

    lines = list(srt.parse(data))

    lyrics = [line.content for line in lines]
    lyrics = [line.replace(",","") for line in lyrics]

    if style is not None:
        lyrics = [line + ', ' + style for line in lyrics]

    pprint(lyrics)

    offsets = [datetime.timedelta(0.)] + [line.end for line in lines]

    num_interpolation_steps = [int(np.ceil((end-start).total_seconds() * fps)) for start, end in zip(offsets, offsets[1:])]
    return lyrics, num_interpolation_steps


def export_to_video(folder):
    frameSize = (img_width, img_height)
    out_file = folder / "output_video.mp4"
    out = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'avc1'), fps, frameSize)
    files = sorted(glob.glob(f'{folder}/*.png'), key=lambda x: int(Path(x).stem))
    for filename in files:
        img = cv2.imread(str(filename))
        out.write(img)

    out.release()
    print(f"Saved video to {out_file}")

def magic(folder, prompts, num_frames):
    
    # Enable mixed precision
    # (only do this if you have a recent NVIDIA GPU)
    keras.mixed_precision.set_global_policy("mixed_float16")

    # Instantiate the Stable Diffusion model
    model = keras_cv.models.StableDiffusion(jit_compile=True, img_width=img_width, img_height=img_height)

    noise = tf.random.normal((img_height // 8,  img_width // 8, 4), seed=seed)

    # repeat last frame
    num_frames += [0]
    prompts += [prompts[-1]]

    enc_prev = tf.squeeze(model.encode_text(prompts[0]))
    i = 0
    all_images = []
    with tqdm.tqdm(total=sum(num_frames)) as pbar:
        for j, (prompt, steps) in enumerate(zip(prompts[1:], num_frames)):
            
            enc = tf.squeeze(model.encode_text(prompt))
            
            interp_encodings = tf.linspace(enc_prev, enc, steps)
            enc_prev = enc
            
            for batch in range(0, steps, batch_size):
                index = tf.range(batch, min(batch + batch_size, steps))
                encoding_batch = tf.gather(interp_encodings, index)
                images_batch = encoding_batch
                images = model.generate_image(
                    encoding_batch,
                    batch_size=len(encoding_batch),
                    num_steps=25,
                    diffusion_noise=noise,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                )

                images_batch = [Image.fromarray(image) for image in images]
                [img.save(folder / f'{_i}.png') for _i, img in zip(range(i, i + len(images_batch)), images_batch)]
                i += len(images_batch)
                
                all_images += images_batch

                pbar.update(len(encoding_batch))
    
    return all_images

if __name__ == '__main__':
    audio_file = Path("i_am_the_walrus.mp3")
    save_folder = Path("output")
    save_folder.mkdir(exist_ok=False, parents=False)

    lyrics_file = extract_lyrics(audio_file, save_folder)

    lyrics, num_frames = read_lyrics(lyrics_file, fps)
    prompts = lyrics

    images = magic(save_folder, prompts, num_frames)

    export_to_video(save_folder)
    assert len(images) == sum(num_frames)