import re

def extract_response_features(text):
    valid_emotions = [
        "happy", "sad", "curious", "anxious", "excited", 
        "sleepy", "loving", "surprised", "confused", "content"
    ]

    emotion_pattern = r'\((%s)\)' % '|'.join(valid_emotions)

    emotions = re.findall(emotion_pattern, text)
     
    motions = re.findall(r'\{([^}]+)\}', text)
    sounds = re.findall(r'<([^>]+)>', text)

    # emojis = re.findall(r'[\U0001F600-\U0001F64F]', text)  

    return {
        "motions": motions,
        "sounds": sounds,
        "emotions": emotions   
    }

