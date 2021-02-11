from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

def getVideoWithSubtitles(link):
    getVideo(link)
    getSubtitles(link[(link.find("="))+1:])

def getVideo(video_url):
    YouTube(video_url).streams.first().download('/Users/umakant/Desktop/FYP/data')

def getSubtitles(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    
    with open('subtitle_32.txt', 'w') as f:
        for item in transcript_list:
            f.write("%s\n" % item)

video_link = "https://www.youtube.com/watch?v=VdLf4fihP78"
getVideoWithSubtitles(video_link)