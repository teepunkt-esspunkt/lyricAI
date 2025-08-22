import lyricAI_functions
import os

lyricAI_functions.combine_all(lyricAI_functions.LyricsOutputDir, lyricAI_functions.LyricsOutputFile)

with open(os.path.join(lyricAI_functions.script_dir, lyricAI_functions.LyricsOutputFile), "w", encoding="utf-8") as f:
    for x in f:
        if "Contributor" in x.lower():
            x = ""