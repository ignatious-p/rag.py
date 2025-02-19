# rag.py

rag.py: LLM chatbot with RAG, response streaming and chat history in one file


## Quickstart

Assuming you have the required dependancies, simply download the file `rag.py` and run it as follows:

```bash
uv run rag.py
```

Congratulations, you now are running a completely private LLM locally with RAG, response streaming and chat history, _without paying a **penny** to OpenAI OR uploading **anything** to the cloud_.
## Dependencies
NOTE: THIS SCRIPT IS ONLY COMPATIBLE WITH UBUNTU/DEBIAN-BASED LINUX! MACOS AND WINDOWS ARE CURRENTLY _UNSUPPORTED_!  

You will need [uv](https://github.com/astral-sh/uv) to run this script as a single file.

You will need Python >= 3.10.



## Installation

1. Clone this git repository

2. `cd` into the directory where the repository was cloned into

3. Run the following command (to run the file as a script). You don't need to setup a virtual environment or anything for this as `uv` creates temporary virtual environments for scripts:

```bash
uv run rag.py
```

This may take a while on the first run...


4. To develop this script, you need to create a virtual environment:

```bash
uv venv
```


    