#!/usr/bin/env python3
"""
Example: Using Auto-Download from Hugging Face
This example demonstrates how to automatically download models from Hugging Face
"""

import torch
import soundfile as sf

# ref parent directory
import sys
sys.path.append('../')

from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline


def main():
    """Demonstrate auto-download functionality"""
    print("🚀 TaDiCodec Auto-Download Example")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example 1: Auto-download TaDiCodec
    print("\n1️⃣ Loading TaDiCodec with auto-download...")
    try:
        # This will automatically download from Hugging Face if not found locally
        tadicodec = TaDiCodecPipline.from_pretrained(
            ckpt_dir="amphion/TaDiCodec", device=device, auto_download=True
        )
        print("✅ TaDiCodec loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load TaDiCodec: {e}")
        return

    # Example 2: Auto-download TTS Pipeline
    print("\n2️⃣ Loading TTS Pipeline with auto-download...")
    try:
        # This will automatically download both TaDiCodec and LLM from Hugging Face
        tts_pipeline = TTSInferencePipeline.from_pretrained(
            tadicodec_path="amphion/TaDiCodec",
            llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
            device=device,
            auto_download=True,
        )
        print("✅ TTS Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load TTS Pipeline: {e}")
        return

    # Example 3: Test speech reconstruction
    print("\n3️⃣ Testing speech reconstruction...")
    try:
        # Use the test audio files
        prompt_text = "In short, we embarked on a mission to make America great again, for all Americans."
        target_text = "But to those who knew her well, it was a symbol of her unwavering determination and spirit."

        prompt_speech_path = "./test_audio/trump_0.wav"
        speech_path = "./test_audio/trump_1.wav"

        # Reconstruct audio with text guidance
        rec_audio = tadicodec(
            text=target_text,
            speech_path=speech_path,
            prompt_text=prompt_text,
            prompt_speech_path=prompt_speech_path,
        )

        # Save the reconstructed audio
        output_path = "./auto_download_test_rec.wav"
        sf.write(output_path, rec_audio, 24000)
        print(f"✅ Speech reconstruction completed! Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Speech reconstruction failed: {e}")

    # Example 4: Test TTS generation
    print("\n4️⃣ Testing TTS generation...")
    try:
        # Generate speech with code-switching
        audio = tts_pipeline(
            text="但是 to those who 知道 her well, it was a 标志 of her unwavering 决心 and spirit.",
            prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
            prompt_speech_path="./test_audio/trump_0.wav",
        )

        # Save the generated audio
        output_path = "./auto_download_test_tts.wav"
        sf.write(output_path, audio, 24000)
        print(f"✅ TTS generation completed! Saved to: {output_path}")

    except Exception as e:
        print(f"❌ TTS generation failed: {e}")

    print("\n🎉 Auto-download example completed!")
    print("💡 Models are now cached locally for faster future use.")


if __name__ == "__main__":
    main()
