# SeGaTag - Semantic Gaze Tagging
Repo with code for semantically tagging gaze data and doing some analyses.
Currently, it is organised as follows:

- `convert_osie.py`: Loads .h5 files one by one and finds the relevant gaze for all the images used as stimuli. The images are form the OSIE paradigm. It saves a .csv file per session that contains information about the stimuli and the tagged gaze. Some noteworthy info: 
1. Which semantic dimensions participants looked at
2. Which semantic dimensions are present in the stimulus
3. The relative salience (high #1, low #2 -> high #3, and vice versa).

Also some other useful info, like the number of consecutive and total NaNs.
- `sem_analysis.py`: Loads .csv files from `convert_osie.py`, compiles a nice dictionary that averages over sessions or participants, and plots spider charts. These charts are a "fingerprint" of participants.

- `stats_analysis.py`: WiP, does some statistical testing on the data gained from the previously described parts. Concretely, are individuals significantly different? Or, are individuals consistent across sessions?


## Semantic label categories
These are the categories of the OSIE labels, as described in https://jov.arvojournals.org/article.aspx?articleid=2193943.
The [osieLabels] folder contains a .mat file for each of the photos in the OSIE set. Each dimension of the .mat file contains a matrix with
pixel maps for the categories described below. Loaded by 'load_sem_data' in 'utils.py'.
| Index | Category     | Description                                                                                             |
|-------|--------------|---------------------------------------------------------------------------------------------------------|
| 0     | Face         | Back, profile, and frontal faces.                                                                       |
| 1     | Emotion      | Faces with obvious emotions.                                                                            |
| 2     | Touched      | Objects touched by a human or animal in the scene.                                                      |
| 3     | Gazed        | Objects gazed upon by a human or animal in the scene.                                                   |
| 4     | Motion       | Moving/flying objects, including humans/animals with meaningful gestures.                               |
| 5     | Sound        | Objects producing sound (e.g., a talking person, a musical instrument).                                 |
| 6     | Smell        | Objects with a scent (e.g., a flower, a fish, a glass of wine).                                         |
| 7     | Taste        | Food, drink, and anything that can be tasted.                                                           |
| 8     | Touch        | Objects with a strong tactile feeling (e.g., a sharp knife, a fire, a soft pillow, a cold drink).       |
| 9     | Text         | Digits, letters, words, and sentences.                                                                  |
| 10    | Watchability | Man-made objects designed to be watched (e.g., a picture, a display screen, a traffic sign).            |
| 11    | Operability  | Natural or man-made tools used by holding or touching with hands.                                       |
