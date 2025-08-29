import torch

# ref parent directory
import sys
sys.path.append('../')

from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline


# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create pipeline
    pipeline = TTSInferencePipeline.from_pretrained(
        tadicodec_path="amphion/TaDiCodec",
        llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-3B",
        device=device,
    )
    # Inference on single sample
    audio = pipeline(
        text="But to those who know her well, it was a symbol of her unwavering determination and spirit.",
        prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
        prompt_speech_path="./test_audio/trump_0.wav",
    )

    # audio = pipeline(
    #     text="寡妇马华莎，光汉贾家嘉。马华莎脸上麻，贾家嘉独眼瞎，两人登记成了家。贾家嘉不嫌马华莎麻，马华莎不嫌贾家嘉瞎。",
    #     prompt_text="皇上登基都已经大半年了，可是，这每个月进后宫的日子，掰着指头都数的清楚。",
    #     prompt_speech_path="./use_examples/test_audio/zh_005.wav",
    # )

    # Save audio
    import soundfile as sf

    sf.write("./lm_tts_output.wav", audio, 24000)
