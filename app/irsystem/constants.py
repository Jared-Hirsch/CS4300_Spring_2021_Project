# Project Information
PROJECT_NAME = "Spotify Genus Recommendations"
NAMES = [["Jared Hirsch", "jsh328"], ["Ronnie Dumesh", "rad338"],
         ["Ravina Patel", "rpp62"], ["Chris Chen", "cmc447"]]

# Parameters
INPUT_QUERY = 'input_query'
REQUERY = 'requery'
LIKED = 'liked'
DISLIKED = 'disliked'
LYRICAL_SIMILARITY = 'lyrical_sim'
AUDIO_SIMILARITY = 'aud_sim'
NUM_SONGS = 'duration'
NUM_RESULTS = 10
AUDIO_FEATURES = [["Acousticness", "auc", "A confidence measure of whether the track is acoustic."],
                  ["Danceability", "dnc", "Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity."],
                  ["Energy", "enr", "A perceptual measure of intensity and activity. Tracks with high energy tend to feel fast, loud, and noisy."],
                  ["Instrumentalness", "ins", "A confidence measure of whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental."],
                  ["Liveness", "liv", "A confidence measure of whether the presence of an audience in the recording. Higher values indicate that the track was likely performed live."],
                  ["Loudness", "loud", "The overall loudness of a track in decibels [dB]."],
                  ["Speechiness", "spch", "Detects the presence of spoken words in a track. Tracks with high values sound more speech-like."],
                  ["Tempo", "temp", "The overall estimated tempo of a track in beats per minute [BPM]."],
                  ["Valence", "val", "Describes how positive the track is. Tracks with high valence sound more happy or euphoric, while tracks with low valence sound more negative."], ]

# Templates
INDEX = 'index.html'
RESULTS = 'results.html'
ERROR_404 = '404.html'

#Spotify Scope
SPOTIFY_SCOPE = 'playlist-modify-private playlist-modify-public'
