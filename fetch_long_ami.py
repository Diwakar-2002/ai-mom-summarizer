import io
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio

def main():
    print("Loading edinburghcstr/ami dataset from Hugging Face to create a 5-minute audio...")
    
    # Stream the dataset to avoid downloading the whole thing
    dataset = load_dataset("edinburghcstr/ami", "ihm", split="validation", streaming=True)
    
    # We will need the actual audio data to concatenate, so we keep decoding enabled
    # But wait, earlier we had a torchcodec error with decoding.
    # We can cast to disable decoding, read bytes with soundfile via BytesIO.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    target_duration = 300  # 5 minutes in seconds
    current_duration = 0.0
    
    audio_arrays = []
    texts = []
    sample_rate = None
    
    print("Fetching chunks...")
    for example in dataset:
        audio_info = example["audio"]
        text = example["text"]
        
        if "bytes" in audio_info and audio_info["bytes"] is not None:
            # Read WAV bytes using soundfile and BytesIO
            data, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        elif "path" in audio_info and audio_info["path"]:
            data, sr = sf.read(audio_info["path"])
        else:
            continue
            
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            # Should not happen in same dataset but just in case
            continue
            
        duration = len(data) / sample_rate
        
        audio_arrays.append(data)
        texts.append(text)
        current_duration += duration
        
        if current_duration >= target_duration:
            break

    print(f"Gathered {len(audio_arrays)} chunks. Total duration: {current_duration:.2f} seconds.")
    
    # Concatenate the audio arrays
    concatenated_audio = np.concatenate(audio_arrays, axis=0)
    
    # Concatenate the text
    full_text = " ".join(texts)
    
    # Save the new 5-minute audio
    audio_filename = "ami_long_sample.wav"
    sf.write(audio_filename, concatenated_audio, sample_rate)
    print(f"Saved {current_duration:.2f}s audio to {audio_filename}")
    
    # Save the ground truth text
    text_filename = "ami_long_ground_truth.txt"
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Saved ground truth text to {text_filename}")

if __name__ == "__main__":
    main()
