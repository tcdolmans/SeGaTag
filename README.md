# SeGaTag - Semantic Gaze Tagging
Repo with code for semantically tagging gaze data and doing some analyses.
Currently, it is organised as follows:

- `convert_osie.py`: Loads .h5 files one by one and finds the relevant gaze for all the images used as stimuli. The images are form the OSIE paradigm. It saves a .csv file per session that contains information about the stimuli and the tagged gaze. Some noteworthy info: 
1. Which semantic dimensions participants looked at
2. Which semantic dimensions are present in the stimulus
3. The relative salience (high #1, low #2 -> high #3, and vice versa).
Also some other useful info, like the number of consecutive and total NaNs.
- `sem_analysis.py`: Loads .csv files from `convert_osie.py`, compiles a nice dictionary that averages over sessions or participants, and plots spider charts. These charts are a "fingerprint" of participants.

- `stats_analysis.py`: WiP, does some statistical testing on the data gained from the previously described parts. Concretely, are individuals significatnly different? Or, are individuals consistent across sessions?
