# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chromadb",
#     "ollama",
#     "pypdf",
# ]
# ///

from asyncio import CancelledError
import shutil
import subprocess
import platform
import sys
import chromadb
import pypdf

from ollama import chat

# For simplicity, we use an in-memory Chroma client.
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="pdf_documents")

CHAT_MODEL = "granite3.1-dense:2b"
# CHAT_MODEL = "qwen2.5:3b"
# EMBEDDING_MODEL = "granite-embedding:30m"

PDF_FILE = "test_dunning_kruger.pdf"  # Predefined PDF file path

# A threshold for deciding if a query is relevant to the PDF context.
# TODO: Tune this value based on the embeddings model
CONTEXT_THRESHOLD = 1.3


# %% PDF Processing Functions


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def split_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Naively split the text into chunks of words.
    TODO: This can be replaced by a more advanced text splitter if needed.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # create overlap between chunks
    return chunks


def index_pdf_to_db(pdf_path: str):
    """Extracts text from the PDF, splits it into chunks, and indexes it into the vector DB."""
    print(f"Indexing PDF file: {pdf_path}")
    text = extract_pdf_text(pdf_path)
    chunks = split_text(text)
    if not chunks:
        print("No text extracted from the PDF.")
        return
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    # Add documents to the collection.
    # TODO: pass in an embedding function.
    collection.add(documents=chunks, ids=ids)
    print(f"Indexed {len(chunks)} text chunks from the PDF.")


# %% Ollama Installation Functions


def check_ollama_installed():
    """Check if Ollama is already installed on the system."""
    try:
        ollama_path = shutil.which("ollama")
        if ollama_path:
            print("Found ollama")
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
    except Exception:
        pass
    return False


def get_system_info():
    """Determine the operating system and architecture."""
    system = platform.system().lower()
    _machine = platform.machine().lower()

    if _machine in ["x86_64", "amd64"]:
        arch = "amd64"
    elif _machine in ["arm64", "aarch64"]:
        arch = "arm64"
    else:
        raise SystemError(f"Unsupported architecture: {_machine}")

    return system, arch


def install_ollama():
    """Main installation function."""
    try:
        system, _arch = get_system_info()

        if system == "linux":
            print("Installing Ollama...")
            install_script = "curl https://ollama.ai/install.sh | sh"
            subprocess.run(install_script, shell=True, check=True)
        else:
            raise SystemError(f"Unsupported operating system: {system}")

        print("\nOllama installation completed successfully!")
        print("You can start using Ollama by running 'ollama' in your terminal.")

    except Exception as e:
        print(f"\nError during installation: {str(e)}")
        sys.exit(1)


# %% Model Download Functions


def check_ollama_models():
    """Check if we have the models we need."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if CHAT_MODEL in result.stdout:
        print(f"Found required model: {CHAT_MODEL}")
        return True
    else:
        return False


def download_ollama_models():
    """Download the models we need."""
    result = subprocess.run(
        ["ollama", "pull", CHAT_MODEL], capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Successfully pulled model: {CHAT_MODEL}")
    else:
        print(f"Error pulling model: {CHAT_MODEL}")


# %% Main

if __name__ == "__main__":
    # Ensure Ollama is installed.
    print("Looking for ollama...")
    if not check_ollama_installed():
        install_ollama()

    # Ensure the required model is available.
    # TODO: Add support for downloading multiple models
    print("Looking for models...")
    if not check_ollama_models():
        download_ollama_models()

    # Index the PDF file into the vector DB.
    try:
        index_pdf_to_db(PDF_FILE)
    except Exception as e:
        print(f"Error indexing PDF: {e}")

    print("All set. Running model...")

    messages: list[dict[str, str]] = []

    while True:
        try:
            query = input("User: ")

            # Query the vector database for context related to the user's query.
            context = ""
            try:
                query_results = collection.query(query_texts=[query], n_results=1)

                # Assuming that lower distance means higher similarity.
                if query_results["distances"] and query_results["distances"][0]:
                    similarity = query_results["distances"][0][0]

                    # for debug purposes
                    print(f"similarity: {similarity}")

                    if similarity < CONTEXT_THRESHOLD:
                        context = query_results["documents"][0][0]

                        print("using PDF context for query.")

                        # for debug purposes
                        print(f"extra context: {context}")

            except Exception as db_err:
                print(f"Error querying vector DB: {db_err}")

            # If context is found, prepend it to the query.
            if context:
                combined_query = (
                    f"Relevant PDF context:\n{context}\n\nUser Query: {query}"
                )
            else:
                combined_query = query

            # Append the query message.
            messages.append({"role": "user", "content": combined_query})

            # Stream the response from the model.
            stream = chat(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
            )

            print("Model: ", end="")
            response = ""
            for chunk in stream:
                chunk_text = chunk["message"]["content"]
                response += chunk_text
                print(chunk_text, end="", flush=True)

            print("")
            messages.append({"role": "assistant", "content": response})

        except (CancelledError, EOFError):
            break

        except KeyboardInterrupt:
            print("Leaving...")
            break
