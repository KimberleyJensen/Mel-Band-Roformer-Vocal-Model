import time
import numpy as np
import torch
import sys
import torch.nn as nn


def get_model_from_config(model_type, config):
    if model_type == 'mel_band_roformer':
        from models.mel_band_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model


def demix_track(config, model, mix, device, first_chunk_time=None):
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

            i = 0
            total_length = mix.shape[1]
            num_chunks = (total_length + step - 1) // step

            if first_chunk_time is None:
                start_time = time.time()
                first_chunk = True
            else:
                start_time = None
                first_chunk = False

            while i < total_length:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

                if first_chunk and i == 0:
                    chunk_start_time = time.time()

                x = model(part.unsqueeze(0))[0]
                result[..., i:i+length] += x[..., :length]
                counter[..., i:i+length] += 1.
                i += step

                if first_chunk and i == step:
                    chunk_time = time.time() - chunk_start_time
                    first_chunk_time = chunk_time
                    estimated_total_time = chunk_time * num_chunks
                    print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
                    first_chunk = False

                if first_chunk_time is not None and i > step:
                    chunks_processed = i // step
                    time_remaining = first_chunk_time * (num_chunks - chunks_processed)
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds")
                    sys.stdout.flush()

            print()
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}, first_chunk_time
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}, first_chunk_time
