import os
import requests

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded successfully: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e

def main():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    downloads = [
        {
            "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
            "dest": os.path.join(models_dir, "silero_vad.onnx")
        },
        {
            "url": "https://huggingface.co/FunAudioLLM/CosyVoice-300M/resolve/main/campplus.onnx",
            "dest": os.path.join(models_dir, "campplus.onnx")
        },
        {
            "url": "https://huggingface.co/spaces/coqui/xtts/resolve/main/examples/female.wav",
            "dest": os.path.join(models_dir, "spk1.wav")
        },
        {
            "url": "https://huggingface.co/spaces/coqui/xtts/resolve/main/examples/male.wav",
            "dest": os.path.join(models_dir, "spk2.wav")
        }
    ]
    
    for item in downloads:
        download_file(item["url"], item["dest"])

if __name__ == "__main__":
    main()
