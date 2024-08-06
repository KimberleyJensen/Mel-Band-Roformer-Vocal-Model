A [Mel-Band-Roformer Vocal model](https://arxiv.org/abs/2310.01809). This model performs slightly better than the paper equivalent due to training with more data.

# How to use

Download the model - https://drive.google.com/file/d/19vVn6Yn_ppiELOvMOR5m-CmNT_FCiDs3/view?usp=sharing

Install requirements - `pip install -r requirements.txt`

Inference - `python inference.py --config_path configs/config_vocals_mel_band_roformer.yaml --model_path melbandroformer.ckpt --input_folder songsfolder --store_dir outputsfolder`

The model will perform inference on every .wav file inside the --input_folder and save them to the --store_dir folder.

[num_overlap](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/41d04ae1c8ea89261b488e90953192efe650fa4f/configs/config_vocals_mel_band_roformer.yaml#L38) - Increasing this value can improve the quality of the outputs due to helping with artifacts created when putting the chunks back together. This will make longer inference times longer (you don't need to go higher than 8)

[chunk_size](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/41d04ae1c8ea89261b488e90953192efe650fa4f/configs/config_vocals_mel_band_roformer.yaml#L39) - The length on audio input into the model (default is 352800 which is 8 seconds, 352800 was also used to train the model)

# Thanks to

[lucidrains](https://github.com/lucidrains) For [implementing the paper](https://github.com/lucidrains/BS-RoFormer) (and all his open source work)

[ZFTurbo](https://github.com/ZFTurbo) for releasing [training code](https://github.com/ZFTurbo/Music-Source-Separation-Training) which was used to train the model

Ju-Chiang Wang, Wei-Tsung Lu, Minz Won (ByteDance AI Labs) - The authors of the Mel-Band RoFormer paper

# How to help

If you want to contribute GPU access for me to train a bigger model (this model did not overfit) please contact me at KimberleyJensenOx1@gmail.com. A 48GB GPU will be required due to high VRAM usage. You can also contribute by adding to my dataset which is used to train the model. 




