# Downward Translate

This code was copied from https://github.com/aibasel/downward/tree/main/src/translate and lightly modified to be used with the scoping library. The main changes were:

1. Using absolute imports
2. A few type hints
3. Added scoping functionality


The main point of interaction is [translate_and_scope.py](translate_and_scope.py). It can be called as a script, or the function `main_from_other_script` can be called. The options/kwargs to pass in are specified in [options.py](options.py).