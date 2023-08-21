# SeGaTag - Semantic Gaze Tagging
Repo with code for semantically tagging gaze data and doing some analyses. See the `environment.yml` file for dependencies.
Currently, it is organised as follows:

## Files
- `convert_osie.py`: Loads .h5 files one by one and finds the relevant gaze for all the images used as stimuli. The images are form the OSIE paradigm. It saves a .csv file per session that contains information about the stimuli and the tagged gaze. Some noteworthy info: 
1. Which semantic dimensions participants looked at
2. Which semantic dimensions are present in the stimulus
3. The relative salience (high #1, low #2 -> high #3, and vice versa).
Also some other useful info, like the number of consecutive and total NaNs.


- `sem_analysis.py`: Loads .csv files from `convert_osie.py`, compiles a nice dictionary that averages over sessions or participants, and plots spider charts. These charts are a "fingerprint" of participants. It also compiles a master list that can be sliced to select subsets of the data by participant, img number, or even the content of the images. See below table for help with slicing the master. 


- `stats_analysis.py`: *WiP*, does some statistical testing on the data gained from the previously described parts. Need more data for this to make sense. Analysis scripts should be adapted to work with the master list from `sem_analysis.py`.

## Semantic Label Categories
These are the categories of the OSIE labels, as described in https://jov.arvojournals.org/article.aspx?articleid=2193943.
The `osieLabels` folder contains a .mat file for each of the photos in the OSIE set. Each dimension of the .mat file contains a matrix with
pixel maps for the categories described below. Loaded by ``load_sem_data`` in `utils.py`.
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

## Folder Structure
In order for this all to work, ensure you have the following file structure:
src
 - osieData (contains all h5 files to be used, NOT in this repo for privacy reasons)
 - osieImgs (contains all OSIE images as .jpg, NOT in the repo for copyright reasons)
 - osieLabels (contains all labels for the OSIE images, in this repo)
 - SeGaTag (contains all code in this repo)
 - semPreProc (will contain outputs after running `convert_osie.py`)
    - raw (will contain raw outputs if the flag is set `True`)
