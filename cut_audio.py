import os
import subprocess
import imageio_ffmpeg

def main():
    audio_file = "alameda_05fd2fe2-ce9f-48cf-8acc-c0a49d6a8067.mp3"
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return

    # Create chunks directory
    os.makedirs("chunks", exist_ok=True)

    # Get FFmpeg executable path
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using FFmpeg: {ffmpeg_exe}")
    except Exception as e:
        print(f"Error getting FFmpeg executable: {e}")
        return

    # Slicing parameters: 5 chunks of 15 minutes (900 seconds) each
    chunk_duration = 900  # seconds
    num_chunks = 5

    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_file = os.path.join("chunks", f"chunk_{i+1}.mp3")
        
        print(f"Slicing chunk {i+1}/{num_chunks}: {start_time}s to {start_time + chunk_duration}s...")
        
        # Build FFmpeg command for stream copying (extremely fast, no re-encoding)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", audio_file,
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-acodec", "copy",
            output_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully created: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk {i+1}: {e.stderr}")
            return

    print("All chunks created successfully!")

if __name__ == "__main__":
    main()
