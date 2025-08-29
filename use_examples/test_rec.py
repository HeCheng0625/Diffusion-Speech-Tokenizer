import torch
import soundfile as sf

# ref parent directory
import sys
sys.path.append('../')

from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline

if __name__ == "__main__":

    print("--------------------------------")
    print("Test reconstruction without prompt")
    print("--------------------------------")

    # Test reconstruction without prompt
    # We use prefix 3s of the target audio as the prompt
    # you can modify the decoding logic in models/tts/tadicodec/inference_tadicodec.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = TaDiCodecPipline.from_pretrained(ckpt_dir="amphion/TaDiCodec", device=device)

    # Text of the audio
    text = "离别没说再见，你是否心酸。"
    # Input audio path
    speech_path = "./test_audio/sing_001.wav"
    # Generate reconstructed audio with 3s prefix as prompt
    rec_audio = pipe(text=text, speech_path=speech_path, cfg_scale=2.0, n_timesteps=32)
    sf.write("./sing_rec.wav", rec_audio, 24000)

    print("--------------------------------")
    print("Test reconstruction with prompt")
    print("--------------------------------")

    # Text of the prompt audio
    prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
    # Text of the target audio
    target_text = "But to those who knew her well, it was a symbol of her unwavering determination and spirit."

    # Input audio path of the prompt audio
    prompt_speech_path = "./test_audio/trump_0.wav"
    # Input audio path of the target audio
    speech_path = "./test_audio/trump_1.wav"

    rec_audio = pipe(
        text=target_text,
        speech_path=speech_path,
        prompt_text=prompt_text,
        prompt_speech_path=prompt_speech_path,
        cfg_scale=2.0,
        n_timesteps=32,
    )
    sf.write("./trump_rec.wav", rec_audio, 24000)

    print("--------------------------------")
    print("Test extract tadicodec tokens")
    print("--------------------------------")

    tokens = pipe(text=target_text, speech_path=speech_path, return_code=True)
    print(tokens)
