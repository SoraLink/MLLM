import os
import subprocess
from time import perf_counter
from typing import Optional, List, Sequence

import einops
import jax
import gin
import librosa
import nest_asyncio
import matplotlib.pyplot as plt
import pylab
import numpy as np
from PIL import Image
import tensorflow as tf
import gradio as gr
from flax.core import freeze
import seqio
from absl import logging
from PIL import ImageColor, ImageOps
import cv2


# Uncomment to log compilations for jitted funtions
# environ["JAX_LOG_COMPILES"] = "1"

# On by default in train.py and eval.py
jax.config.update("jax_parallel_functions_output_gda", True)

# Hack for https://github.com/scikit-video/scikit-video/issues/154, not sure
# how were avoiding this issue before but this fix works me
import numpy

numpy.float = numpy.float64
numpy.int = numpy.int_

from t5x import utils
from t5x import partitioning

from t5x.examples.unified_io.data import tasks
from t5x.examples.unified_io.data.data_utils import (
    get_default_vocabulary,
    resize_and_pad_default,
    MODALITY_EXTRA_ID_N_FRAMES,
)

from t5x.examples.unified_io import modality_processing
from t5x.examples.unified_io import utils as uio_utils
from t5x.examples.unified_io import config
from t5x.examples.unified_io import decoding
from t5x.examples.unified_io.metrics.utils import (
    extract_bboxes_from_text,
    extract_actions_from_prediction,
    reconstruct_vima_action,
    extract_points_from_text,
)

import demo.utils as U

logging.set_verbosity(logging.INFO)

# Needed when running interactively
nest_asyncio.apply()
gin.enter_interactive_mode()


FULL_CKPT_PATH = "xxl-3m"  # Modify to point to your checkpoint
MODEL_TYPE = "xxl"  # Set 'large', 'xl', or 'xxl

assert FULL_CKPT_PATH is not None

vocab = get_default_vocabulary()

# For GPUS
n_gpus = 1
supports_bfloat16 = False  # If your GPUs support this, setting to true to improve performance
model = uio_utils.get_model(MODEL_TYPE, dtype="bfloat16" if supports_bfloat16 else "float32")
partitioner = partitioning.PjitPartitioner(num_partitions=n_gpus)

parameters, param_axes = uio_utils.get_parameters(model, FULL_CKPT_PATH, partitioner)

# Load HIFGAN, which converts the melspectorgram UIO2 outputs into audio waveforms that can be played
USE_HIFIGAN = True
if USE_HIFIGAN:
    from demo.hifigan.models import Generator
    from demo.hifigan.env import AttrDict
    import json
    import torch

    config_file = os.path.join("demo/hifigan/checkpoints", "config.json")
    with open(config_file) as f:
        data = f.read()

    # global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    # global torch_device
    torch_device = torch.device("cpu")

    def load_checkpoint(filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        return checkpoint_dict

    hifigan_generator = Generator(h).to(torch_device)
    state_dict_g = load_checkpoint("demo/hifigan/checkpoints/g_00930000", torch_device)
    hifigan_generator.load_state_dict(state_dict_g["generator"])

    hifigan_generator.eval()
    hifigan_generator.remove_weight_norm()
    print("Complete.")
else:
    hifigan_generator = None


# Define sequence_len, which determines how much to pad the inputs and how many patches to sample
IMG_HISTORY_MAX_FRAMES = 4
AUDIO_HISTORY_MAX_FRAMES = 4

subsample_ratio = 1.0
sequence_len = {
    "is_training": False,
    "text_inputs": 512,
    "text_targets": 512,
    "image_input_samples": int(
        config.IMAGE_INPUT_SIZE[0] // config.IMAGE_INPUT_D * config.IMAGE_INPUT_SIZE[1] // config.IMAGE_INPUT_D * subsample_ratio
    ),
    "image_history_input_samples": int(
        config.IMAGE_HISTORY_INPUT_SIZE[0] // config.IMAGE_HISTORY_INPUT_D
        * config.IMAGE_HISTORY_INPUT_SIZE[1] // config.IMAGE_HISTORY_INPUT_D
        * subsample_ratio
    ),
    "audio_input_samples": int(
        config.AUDIO_INPUT_SIZE[0] // config.AUDIO_INPUT_D * config.AUDIO_INPUT_SIZE[1] // config.AUDIO_INPUT_D  * subsample_ratio
    ),
    "audio_history_input_samples": int(
        config.AUDIO_HISTORY_INPUT_SIZE[0] // config.AUDIO_HISTORY_INPUT_D
        * config.AUDIO_HISTORY_INPUT_SIZE[1] // config.AUDIO_HISTORY_INPUT_D
        * subsample_ratio
    ),
    "num_frames": IMG_HISTORY_MAX_FRAMES,
}


def _fn(
    params,
    batch,
    decoder_params,
    return_all_decodes,
    num_decodes,
    modality,
    length,
    decode_rng,
    top_k,
    top_p,
    temperature,
    repetition_penalty,
    horizontally_pack_inputs,
    alpha=None,
    greedy=True,
    negative_prompt=None
):
    if greedy:
        model._decode_fn = decoding.beam_search
        assert negative_prompt is None
    else:
        model._decode_fn = decoding.temperature_sample
    return model.predict_batch_with_aux(
        params, batch, decoder_params, return_all_decodes, num_decodes, length, modality,
        decode_rng=decode_rng, top_k=top_k, top_p=top_p, temperature=temperature, alpha=alpha,
        repetition_penalty=repetition_penalty, horizontally_pack_inputs=horizontally_pack_inputs,
        negative_prompt=negative_prompt
    )


# A version of the inference function that will lazily be compiled into a partitioned version when called
_partitioned_infer_step = partitioner.partition(
  _fn,
  in_axis_resources=(param_axes,
                     partitioner.data_partition_spec,
                     None, None, None, None),
  out_axis_resources=None,
  # seed (arg 7) temp (arg 10) alpha (arg 13) and negative prompt (15) are left non-static so they don't require re-compilation
  static_argnums=(2, 3, 4, 5, 6, 8, 9, 11, 12, 14)
)

# the partitioned function no longer accepts keywords, so we build a version that does here for convenience
def partitioned_infer_step(
    params,
    batch,
    decoder_params=None,
    return_all_decodes=False,
    num_decodes=1,
    length=None,
    modality="text",
    decode_rng=None,
    top_k=0,
    top_p=1.0,
    temperature=1,
    repetition_penalty=None,
    horizontally_pack_inputs=None,
    alpha=None,
    greedy=True,
    negative_prompt=None
):
    return _partitioned_infer_step(
        params, batch, decoder_params, return_all_decodes, num_decodes,
        modality, length, decode_rng, top_k, top_p, temperature, repetition_penalty,
        horizontally_pack_inputs, alpha, greedy, negative_prompt)


def build_input_dict(input_text, input_image=None, input_audio=None, audio_history=None, image_history=None):
    out = {}

    if input_image is not None:
        image_input = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        image_input, image_input_mask, _ = resize_and_pad_default(image_input, False)
        out["image_inputs"] = image_input
        out["image_input_masks"] = image_input_mask

    if input_text is not None:
        out["text_inputs"] = input_text

    if image_history is not None and not isinstance(image_history, Sequence):
        # Assume image history is a 4D tensor
        assert len(image_history.shape) == 4
        video_tensor = tf.image.convert_image_dtype(image_history, dtype=tf.float32)
        video_tensor, video_mask, _ = resize_and_pad_default(video_tensor, False)
        out["image_history_inputs"] = video_tensor
        out["image_history_input_masks"] = video_mask

    elif image_history is not None and any(x is not None for x in image_history):
        # Image history contains a list of possibly None images
        image_history_inputs = []
        image_history_input_masks = []
        for image in image_history:
            if image is not None:
                img = tf.image.convert_image_dtype(image, dtype=tf.float32)
                img_input, img_input_mask, _ = resize_and_pad_default(img, False, is_history=True)
            else:
                img_input = tf.zeros(
                    [config.IMAGE_HISTORY_INPUT_SIZE[0], config.IMAGE_HISTORY_INPUT_SIZE[1], 3], dtype=tf.float32)
                img_input_mask = tf.zeros(config.IMAGE_HISTORY_INPUT_SIZE, dtype=tf.int32)

            image_history_inputs.append(img_input)
            image_history_input_masks.append(img_input_mask)

        out["image_history_inputs"] = np.stack(image_history_inputs)
        out["image_history_input_masks"] = np.stack(image_history_input_masks)

    # Audio has the same pre-processing for history and input, so we batch them
    all_audio = []
    if input_audio is not None:
        all_audio.append(input_audio[None, :, :])

    if audio_history is not None:
        all_audio.append(audio_history)

    if all_audio:
        print([x.shape for x in all_audio])
        all_audio = tf.concat(all_audio, 0)
        all_audio = tf.transpose(all_audio, perm=[0, 2, 1])

        audio_mask = tf.cast(all_audio != 0, tf.float32)
        all_audio = tf.math.log(tf.clip_by_value(all_audio, 1e-5, 1e5))
        all_audio = all_audio * audio_mask
        audio_mask = tf.cast(audio_mask, tf.int32)
        all_audio = tf.expand_dims(all_audio, -1)

        if input_audio is not None:
            out["audio_inputs"] = all_audio[0]
            out["audio_input_masks"] = audio_mask[0]
            all_audio = all_audio[1:]
            audio_mask = audio_mask[1:]

        if audio_history is not None:
            out["audio_history_inputs"] = all_audio
            out["audio_history_input_masks"] = audio_mask

    return out


def build_batch(
    input_text,
    input_image=None,
    input_audio=None,
    image_history=None,
    audio_history=None,
    return_resized_input_image=False
):
    """Builds a size one batch of preprocessed inputs that can be passed to the model

    input_text: Input string
    input_audio: Input spectrogram
    input_image: Input image array
    audio_history: 3D Tensor of spectrograms
    image_history: Either a list of images, or a 4D tensor of images

    ret: A batch of pre-preprocessed data
    """
    # Pre-processing functions for UIO2 are built for tf.data.Datasets, so we convert
    # the inputs into a size one dataset and apply those functions here,
    # and then extract a size one tf.data.Dataset from the dataset at the end
    batch = build_input_dict(input_text, input_image, input_audio, audio_history, image_history)
    resized_input_image = batch.get("image_inputs") if return_resized_input_image else None
    dataset = tf.data.Dataset.from_tensors(batch)
    dataset = modality_processing.unified_io_preprocessor(
        dataset, modality_processing.OUTPUT_FEATURES, sequence_len)
    converter = modality_processing.UnifiedIOFeatureConverter()
    dataset = converter(dataset, sequence_len)
    dataset = dataset.batch(1)
    if return_resized_input_image:
        return next(dataset.as_numpy_iterator()), resized_input_image
    return next(dataset.as_numpy_iterator())


# Gradio functionality, including loading videos and building audio spectrograms
INPUT_LEN_BUCKETS = np.array([64, 256, 512, 1024])
N_ROWS = 2
N_PER_ROW = 4
assert IMG_HISTORY_MAX_FRAMES % N_PER_ROW == 0  # for convenience
NUM_DECODES = N_ROWS * N_PER_ROW

DUMMY_NEGATIVE_PROMPT = build_batch("")


def load_audio(path):
    spectrograms = U.load_audio(
        path,
        audio_segment_length=config.AUDIO_SEGMENT_LENGTH,
        spectrogram_length=config.AUDIO_SEGMENT_LENGTH,
        max_audio_length=config.AUDIO_SEGMENT_LENGTH * (AUDIO_HISTORY_MAX_FRAMES + 1),
    )
    return spectrograms


def load_video(path, **kwargs):
    if path.endswith(".wav"):
        gr.Warning(f"Input is audio file {path}")
        return None, load_audio(path)
    frames, spectrograms = U.load_video(path, **kwargs)
    return frames, spectrograms


def compute_non_masked_input_tokens(example):
    """Compute the number of non-masked tokens in the input"""
    encoder_len = 0
    for key, v in modality_processing.get_input_modalities().items():
        mask = example[f"inputs/{key}/mask"]
        assert mask.shape[0] == 1
        mask = mask[0]
        seq_len = v.get_static_sequence_len()
        if seq_len is not None:
            # History encoding compress each valid frame into `seq_len` tokens
            valid_frames = tf.reduce_any(tf.reshape(mask > 0, [mask.shape[0], -1]), -1)
            n_frames = tf.reduce_sum(tf.cast(valid_frames, tf.int32))
            encoder_len += n_frames * seq_len
        else:
            encoder_len += mask.sum()
    return encoder_len


def _scale_resize(image_np, desired_output_size):
    desired_height, desired_width = desired_output_size
    height, width = image_np.shape[:2]
    scale_factor = min(desired_height / height, desired_width / width)
    scaled_height = int(height * scale_factor)
    scaled_width = int(width * scale_factor)
    image_pil = Image.fromarray(image_np).resize((scaled_width, scaled_height), Image.BILINEAR)
    return np.array(image_pil)


# Main Gradio inference functions
def run_inference(
    output_modality,
    random_seed,
    top_k,
    top_p,
    temperature,
    guidance_scale,
    repetition_penalty,
    n_outputs,
    negative_prompt,
    input_decoding,
    bbox_annotate_image,
    input_text,
    input_image,
    input_video,
    input_audio,
    *image_histories,
):
    if input_text is None or input_text == "":
        print(gr.Warning('No prompt specified, predictions will be random!'))
        input_text = ""
    output_modality = output_modality.lower()
    # Prefix to automatically add to input text
    prefix = dict(image="[Image] [S] ", audio="[Audio] [S] ", text="[Text] [S] ")[output_modality]

    # Classifier free guidance batch if being used
    if guidance_scale == 0:
        negative_prompt = None
    if negative_prompt is not None:
        negative_prompt_batch = build_batch(prefix + negative_prompt)
    else:
        negative_prompt_batch = None

    audio_history = None
    if all(x is None for x in image_histories):
        image_history = None
    else:
        image_history = image_histories

    # For video/audio streams, do we encode the last frame in the input stead of the history
    encode_last_frame = input_image is None

    # Load video if one is give
    audio_history = None
    encode_first_frame_as_input = False
    if input_video is not None:
        if image_histories is not None:
            print(gr.Warning(f"Overriding image histories with video"))
        if isinstance(input_video, str):
            image_history, audio_history = load_video(
                input_video,
                max_frames=IMG_HISTORY_MAX_FRAMES + int(encode_first_frame_as_input),
                audio_segment_length=config.AUDIO_SEGMENT_LENGTH,
                use_audio=input_audio is None,
                target_size=config.IMAGE_HISTORY_INPUT_SIZE,
            )
        else:
            image_history, audio_history = input_video
        # if encode_first_frame_as_input:
        #     image_input = image_history[-1]
        #     image_history = image_history[:-1]
        #     if audio_input is not None:
        #         audio_input = audio_input[-1]
        #         audio_history = audio_history[:-1]
    else:
        audio_from_video = None

    # Load audio
    if input_audio is not None:
        if audio_history is not None:
            print(gr.Warning(f"Overriding video audio with input"))
        audio_history = load_audio(input_audio)
        input_audio = audio_history[-1]
        audio_history = audio_history[:-1]
    else:
        input_audio = None

    # Get int random seed
    if random_seed.strip() == "":
        random_seed = np.random.randint(0, 2 ** 31)
    else:
        try:
            random_seed = int(random_seed)
        except (TypeError, ValueError) as e:
            # Don't crash the whole thing
            random_seed = np.random.randint(0, 2 ** 31)
            print(gr.Warning(f"Setting seed failed {e} Using {random_seed}"))

    stats = dict(
        modality=output_modality, text=f"\"{input_text}\"", k=top_k, p=top_p, temp=temperature, guidance=guidance_scale,
        n_outputs=n_outputs, n_prompt=f"\"{negative_prompt}\"", dec=input_decoding,
        bbox_annotate_image=bbox_annotate_image,
        input_image=input_image.shape if input_image is not None else None,
        input_audio=input_audio.shape if input_audio is not None else None,
        audio_history=audio_history.shape if audio_history is not None else None,
    )
    if isinstance(image_history, Sequence):
        stats["image_history"] = [None if x is None else x.shape for x in image_history]
    else:
        stats["image_history"] = None if image_history is None else image_history.shape

    print("Infer called:")
    print(", ".join(f"{k}={v}" for k, v in stats.items()))

    if input_image is not None:
        # faster communication after decoding the prediction to gradio demo with lower res image
        input_image = _scale_resize(input_image, config.IMAGE_INPUT_SIZE)

    # Input batch
    ex, resized_image_input = build_batch(prefix + input_text, input_image, input_audio, image_history, audio_history,
                                          True)

    # Pick and input length to compress input tokens into
    n_nonmasked_tokens = compute_non_masked_input_tokens(ex)
    input_len_bucket = n_nonmasked_tokens <= INPUT_LEN_BUCKETS
    if any(input_len_bucket):
        input_len = INPUT_LEN_BUCKETS[np.argmax(input_len_bucket)]
    else:
        input_len = None
    print(f"{n_nonmasked_tokens} inputs, using input len bucket of {input_len}")

    num_decodes = n_outputs
    if input_decoding == "Beam":
        greedy = True
    else:
        greedy = False

    if repetition_penalty and not greedy and output_modality == "text":
        repetition_penalty = 0
        print(gr.Info(f"Repetition penalty not supported for sampling, it will be ignored"))

    # Get the predictions
    update_random_seed = gr.Textbox(label=f"Random seed", value=random_seed)
    output_length = dict(image=1024, audio=512, text=512)[output_modality]

    t0 = perf_counter()
    print("Calling predict...")
    predictions = partitioned_infer_step(
        parameters,
        ex,
        return_all_decodes=True,
        num_decodes=num_decodes,
        decode_rng=jax.random.PRNGKey(random_seed),
        modality=output_modality,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        length=output_length,
        alpha=guidance_scale,
        greedy=greedy,
        negative_prompt=negative_prompt_batch
    )[1]

    # Post-process the predictions to get Gradio outputs
    output = []
    extra_outputs = []
    inputs_that_updated = []
    for k, v in predictions.items():
        v.block_until_ready()
    print(f"Got results in {perf_counter() - t0:0.4f}, converting to numpy...")
    t0 = perf_counter()
    if output_modality == "image":
        image = np.array(predictions["image"])
        if image.shape[0] == 1:
            image = image[0]
        for i in range(num_decodes):
            output.append(np.clip(np.array((image[i] + 1) / 2, dtype=np.float32), 0, 1))
    elif output_modality == "text":
        tokens = np.array(predictions["text-tokens"])
        for i in range(num_decodes):
            output.append(vocab.decode(tokens.tolist()[0][i]))  # type: ignore

        if bbox_annotate_image and input_image is not None:
            for i in range(num_decodes):
                # anno_img = input_image.copy()
                anno_img = resized_image_input.numpy() * 255
                decoder_text = output[i]
                # 3d bounding box, not visualize labels so far
                if decoder_text.startswith("<") or decoder_text.endswith(">"):
                    decoder_text = " " + decoder_text + " "

                bboxes, class_names = extract_bboxes_from_text(decoder_text, anno_img.shape)
                if len(bboxes) > 0:
                    anno_img = draw_bboxes(anno_img, bboxes, color=None)
                else:
                    # 2d keypoints
                    points, class_names_points = extract_points_from_text(decoder_text, anno_img.shape)
                    if len(points) > 0:
                        colors = (
                            LOTS_OF_COLORS
                            if len(points) <= len(LOTS_OF_COLORS)
                            else LOTS_OF_COLORS
                                 * (len(points) // len(LOTS_OF_COLORS) + 1)
                        )
                        for p, c in zip(points, colors):
                            anno_img = cv2.circle(  # type: ignore
                                anno_img, tuple(p.astype(np.int32)[::-1]), 5, c, 2
                            )
                extra_outputs.append(anno_img / 255.0)
        else:
            # just to visualize how the input image cropped and resized
            extra_outputs = [input_image] * len(output)

    elif output_modality == "audio":
        audio = np.array(predictions["audio"])[0]
        for i in range(num_decodes):
            if USE_HIFIGAN:
                with torch.no_grad():
                    # 128 256 1 -> 128 256
                    spectrogram = np.array(audio[i] * 3.8312 - 5.0945)[:, :, 0]
                    spectrogram = torch.FloatTensor(spectrogram).to(torch_device)
                    y_g_hat = hifigan_generator(spectrogram[None, :, :])
                    output_audio = y_g_hat.squeeze().cpu().numpy()
            else:
                spectrogram = np.exp(audio[i] * 3.8312 - 5.0945)[:, :, 0]
                output_audio = librosa.feature.inverse.mel_to_audio(  # type: ignore
                    spectrogram,
                    sr=16000,
                    n_fft=1024,
                    hop_length=256,
                    win_length=None,
                    window="hann",
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                    n_iter=32,
                )
            output.append((config.AUDIO_SAMPLING_RATE, output_audio))
    else:
        raise NotImplementedError(output_modality)

    print(f"Done in {perf_counter() - t0:0.4f} seconds")
    output += [None] * (NUM_DECODES - num_decodes)
    extra_outputs += [None] * (NUM_DECODES - num_decodes)
    assert len(output) == NUM_DECODES
    return *output, *extra_outputs, *inputs_that_updated

# To draw output bounding boxes

LOTS_OF_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 0),  # DarkGreen
    (128, 0, 0),  # Maroon
    (255, 192, 203),  # Pink
    (255, 20, 147),  # DeepPink
    (0, 191, 255),  # DeepSkyBlue
    (147, 112, 219),  # MediumPurple
    (60, 179, 113),  # MediumSeaGreen
]

HEX_COLORS = ["#{:02x}{:02x}{:02x}".format(*_) for _ in LOTS_OF_COLORS]


def draw_bboxes(img, bboxes, color=None):
    bboxes = np.array(bboxes, np.int32)
    img = np.copy(img)

    if color is None:
        color = LOTS_OF_COLORS
        if len(bboxes) > len(color):
            color = color * (len(bboxes) // len(color) + 1)
    elif isinstance(color, str):
        # This will automatically raise Error if rgb cannot be parsed.
        color = [ImageColor.getrgb(color)] * len(bboxes)
    elif isinstance(color[0], (int, float, np.integer, np.float)):
        color = [color] * len(bboxes)
    # y1x1y2x2
    for (x1, y1, x2, y2), c in zip(bboxes, color):
        for x in range(x1, x2):
            img[x, y1] = c
            img[x, y2 - 1] = c
        for y in range(y1, y2):
            img[x1, y] = c
            img[x2, y] = c
    return img