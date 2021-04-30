# Project Information
PROJECT_NAME = "Spotify Genus Recommendations"
NAMES = [("Jared Hirsch", "jsh328"), ("Ronnie Dumesh", "rad338"),
         ("Ravina Patel", "rpp62"), ("Chris Chen", "cmc447")]

# Parameters
INPUT_QUERY = 'input_query'
LYRICAL_SIMILARITY = 'lyrical_sim'
AUDIO_SIMILARITY = 'aud_sim'
NUM_SONGS = 'duration'
NUM_RESULTS = 10
AUDIO_FEATURES = [("Acousticness", "auc", "A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic."),
                  ("Danceability", "dnc", "Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable."),
                  ("Energy", "enr", "Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy."),
                  ("Instrumentalness", "ins", "Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content."),
                  ("Liveness", "liv", "Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live."),
                  ("Loudness", "loud", "The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db."),
                  ("Speechiness", "spch", "Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value."),
                  ("Tempo", "temp", "The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration."),
                  ("Valence", "val", "A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)."), ]

# Templates
INDEX = 'index.html'
RESULTS = 'results.html'
ERROR_404 = '404.html'
