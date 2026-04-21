import ollama
import sys

def download_llava():
    print("Initiating download for LLaVA (Multimodal Vision Model)...")
    print("This may take a few minutes depending on your internet connection.")
    try:
        current_digest, bars = '', {}
        for progress in ollama.pull('llava', stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].update(bars[current_digest].total - bars[current_digest].n)

            if digest and digest not in bars:
                bars[digest] = True
                print(f"Downloading layer {digest[7:19]}...")

            if progress.get('status') == 'success':
                print("\n✅ LLaVA model successfully installed and ready!")
                return
    except Exception as e:
        print(f"\n❌ Error pulling model: {e}")
        print("Please ensure your Ollama background server is running.")

if __name__ == "__main__":
    download_llava()
