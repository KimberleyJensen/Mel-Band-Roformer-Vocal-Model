A [Mel-Band-Roformer Vocal model](https://arxiv.org/abs/2310.01809). This model performs slightly better than the paper equivalent due to training with more data.

# How to use

Download the model - https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt

Install requirements - `pip install -r requirements.txt`

Inference - `python inference.py --config_path configs/config_vocals_mel_band_roformer.yaml --model_path melbandroformer.ckpt --input_folder songsfolder --store_dir outputsfolder`

The model will output a vocals and instrumental file for every .wav file inside the --input_folder and save them to the --store_dir folder.

[num_overlap](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/41d04ae1c8ea89261b488e90953192efe650fa4f/configs/config_vocals_mel_band_roformer.yaml#L38) - Increasing this value can improve the quality of the outputs due to helping with artifacts created when putting the chunks back together. This will make inference times longer (you don't need to go higher than 8)

[chunk_size](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/41d04ae1c8ea89261b488e90953192efe650fa4f/configs/config_vocals_mel_band_roformer.yaml#L39) - The length of audio input into the model (default is 352800 which is 8 seconds, 352800 was also used to train the model)

# Thanks to

[lucidrains](https://github.com/lucidrains) for [implementing the paper](https://github.com/lucidrains/BS-RoFormer) (and all his open source work)

[ZFTurbo](https://github.com/ZFTurbo) for releasing [training code](https://github.com/ZFTurbo/Music-Source-Separation-Training) which was used to train the model

Ju-Chiang Wang, Wei-Tsung Lu, Minz Won (ByteDance AI Labs) - The authors of the Mel-Band RoFormer paper

[aufr33](https://github.com/aufr33) + [Anjok](https://github.com/Anjok07) + [bascurtiz](https://github.com/bascurtiz) for helping contribute to the dataset

# How to help

If you would like to contribute GPU access for me to train better models please contact me at KimberleyJensenOx1@gmail.com. A 40GB GPU will be required due to high VRAM usage. You can also contribute by adding to my dataset which is used to train the model. 

# Google colab

https://colab.research.google.com/drive/1tyP3ZgcD443d4Q3ly7LcS3toJroLO5o1?usp=sharing




