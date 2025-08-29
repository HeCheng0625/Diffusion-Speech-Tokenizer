#!/usr/bin/env python3
"""
Test script for auto-download functionality from Hugging Face
"""

import torch
import soundfile as sf
from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.inference_llm_tts import TTSInferencePipeline
from models.tts.llm_tts.inference_mgm_tts import MGMInferencePipeline


def test_tadicodec_auto_download():
    """Test TaDiCodec auto-download from Hugging Face"""
    print("🧪 Testing TaDiCodec auto-download...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # This should automatically download from Hugging Face if not found locally
        pipe = TaDiCodecPipline.from_pretrained(
            ckpt_dir="amphion/TaDiCodec", device=device, auto_download=True
        )
        print("✅ TaDiCodec loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load TaDiCodec: {e}")
        return False


def test_tts_auto_download():
    """Test TTS pipeline auto-download from Hugging Face"""
    print("\n🧪 Testing TTS pipeline auto-download...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # This should automatically download both models from Hugging Face
        pipeline = TTSInferencePipeline.from_pretrained(
            tadicodec_path="amphion/TaDiCodec",
            llm_path="amphion/TaDiCodec-TTS-AR-Qwen2.5-0.5B",
            device=device,
            auto_download=True,
        )
        print("✅ TTS pipeline loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load TTS pipeline: {e}")
        return False


def test_mgm_auto_download():
    """Test MGM pipeline auto-download from Hugging Face"""
    print("\n🧪 Testing MGM pipeline auto-download...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # This should automatically download both models from Hugging Face
        pipeline = MGMInferencePipeline.from_pretrained(
            tadicodec_path="amphion/TaDiCodec",
            mgm_path="amphion/TaDiCodec-TTS-MGM",
            device=device,
            auto_download=True,
        )
        print("✅ MGM pipeline loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load MGM pipeline: {e}")
        return False


def main():
    """Run all auto-download tests"""
    print("🚀 Testing Auto-Download Functionality from Hugging Face")
    print("=" * 60)

    # Test TaDiCodec
    tadicodec_success = test_tadicodec_auto_download()

    # Test TTS pipeline
    tts_success = test_tts_auto_download()

    # Test MGM pipeline
    mgm_success = test_mgm_auto_download()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"TaDiCodec: {'✅ PASS' if tadicodec_success else '❌ FAIL'}")
    print(f"TTS Pipeline: {'✅ PASS' if tts_success else '❌ FAIL'}")
    print(f"MGM Pipeline: {'✅ PASS' if mgm_success else '❌ FAIL'}")

    if all([tadicodec_success, tts_success, mgm_success]):
        print(
            "\n🎉 All tests passed! Auto-download functionality is working correctly."
        )
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
