
import os
from config import VIDEO_DATA_PATH

def get_data():
    audio_batch = []
    video_batch = []
    # audio_files = sorted(os.listdir(AUDIO_DATA_PATH))
    video_files = sorted(os.listdir(VIDEO_DATA_PATH))
    for file_name in video_files:
        try:
            temp = file_name.split(".")[0]
            video_batch.append(temp)
            audio_batch.append(temp)
        except:
            print("SKIPPING: Something is wrong with {} file".format(file_name))
    print("Number of files prepared ", len(video_batch))
    return (video_batch, audio_batch)