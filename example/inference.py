import librosa

from bigvganinference import BigVGANHFModel, BigVGANInference

# -- model loading ---
model = BigVGANInference.from_pretrained(BigVGANHFModel.V2_22KHZ_80BAND_FMAX8K_256X, use_cuda_kernel=False)
del model

# load using hf uri
model = BigVGANInference.from_pretrained("nvidia/bigvgan_v2_22khz_80band_fmax8k_256x", use_cuda_kernel=False)

# -- inference ---
wav_path = "example/example.wav"
wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True)  # wav is np.ndarray with shape [T_time] and values in [-1, 1]

# Note this implemntation is a wrapper around the get_mel_spectrogram function
# additional audio preprocessing is done to ensure the input is in the correct format
mel = model.get_mel_spectrogram(wav)

# note that torch.inference_mode() is used internally
wav_gen = model(mel)

wav_gen_float = wav_gen.squeeze(0).cpu()  # wav_gen is FloatTensor with shape [1, T_time]
wav_gen_int16 = (wav_gen_float * 32767.0).numpy().astype("int16")  # wav_gen is now np.ndarray with shape [1, T_time] and int16 dtype
