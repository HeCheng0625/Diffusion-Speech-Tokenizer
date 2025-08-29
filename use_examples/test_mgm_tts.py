import torch

# ref parent directory
import sys
sys.path.append('../')

from models.tts.llm_tts.inference_mgm_tts import MGMInferencePipeline


# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create pipeline
    pipeline = MGMInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        mgm_path="amphion/TaDiCodec-TTS-MGM",
        device=device,
    )

    # Inference on single sample
    audio = pipeline(
        text="但是 to those who 知道 her well, it was a 标志 of her unwavering 决心 and spirit.",  # code-switching
        prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
        prompt_speech_path="./test_audio/trump_0.wav",
    )

    # Save audio
    import soundfile as sf

    sf.write("./mgm_tts_output.wav", audio, 24000)
