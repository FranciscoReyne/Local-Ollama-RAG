# Local-Ollama-RAG
Let's go through setting up Retrieval-Augmented Generation (RAG) with Ollama, specifically tailored for a Windows user who wants to integrate this into a Godot game project.

**Overview**

We'll be building a local RAG system.  The core idea is to allow your Godot game to ask questions of a local language model (via Ollama), where the model's responses are informed by a specific set of documents you provide.

**1. Environment Preparation (Windows):**

*   **Install Ollama:** Download the Windows installer for Ollama from [https://ollama.com/](https://ollama.com/) and run it.  Ollama provides a simple way to run language models locally.
*   **Download a Language Model:** In a command prompt or PowerShell terminal, run `ollama pull <model_name>`.  Good starting points are `llama2`, `mistral`, or `codellama`. Choose one that suits your resource limitations (smaller models may be faster). Example: `ollama pull llama2`
*   **Install Python and Libraries:**  We'll use Python for the RAG logic.  First, install Python from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/). Make sure you select "Add Python to PATH" during installation to make it easier to access.
*   **Create a Virtual Environment:** Open a command prompt or PowerShell terminal and navigate to your project directory. Then, create and activate a virtual environment:

    ```bash
    python -m venv rag_godot
    rag_godot\Scripts\activate
    ```

*   **Install Required Libraries:**

    ```bash
    pip install langchain chromadb sentence-transformers
    ```

    These libraries are essential for RAG:

    *   `langchain`: Framework for building language-based applications (RAG pipelines).
    *   `chromadb`:  A lightweight, in-memory vector database for storing and searching document embeddings.
    *   `sentence-transformers`: Provides models for generating sentence embeddings.

**2. Data Preparation:**

*   **Gather Your Documents:** Collect the documents you want your chatbot to use as its knowledge base. These can be `.txt`, `.pdf`, `.md`, or other formats. Place these files within your Godot project folder or a dedicated data directory.
*   **Load and Split Documents:** Use Langchain to load your documents and split them into manageable chunks for embedding and retrieval.

    ```python
    from langchain.document_loaders import TextLoader  # For text files
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Load documents
    loader = TextLoader("data/my_documents.txt")  #  Adjust path as needed
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    ```

**3. Embedding Generation:**

*   **Choose an Embedding Model:**  `sentence-transformers` provides pre-trained models. `all-mpnet-base-v2` is a popular choice.

    ```python
    from langchain.embeddings import HuggingFaceEmbeddings

    model_name = "all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    ```

*   **Generate Embeddings:** Use the chosen model to create embeddings (vector representations) for each document chunk.

**4. Vector Database Setup (ChromaDB):**

*   **Initialize ChromaDB:**  Create a ChromaDB instance to store and query your embeddings.

    ```python
    import chromadb
    from langchain.vectorstores import Chroma

    # Initialize ChromaDB - For memory storage (testing)
    client = chromadb.Client()

    # For disk persistence (recommended for larger projects)
    # client = chromadb.PersistentClient(path="path/to/chromadb_data")

    # Create the vector store
    db = Chroma.from_documents(chunks, embeddings, client=client, collection_name="my_collection")
    ```
    **Important:**  For persistent storage across Godot game sessions, use `chromadb.PersistentClient()` and specify a path within your Godot project folder (e.g., `user://chromadb_data`).  The `user://` path is Godot's designated directory for user-specific data.

**5. Langchain Integration with Ollama:**

*   **Create an Ollama LLM Wrapper:** Tell Langchain how to interact with Ollama.

    ```python
    from langchain.llms import Ollama

    # Specify the model you downloaded from Ollama
    ollama_llm = Ollama(model="llama2")  # Replace with your model
    ```

**6. RAG Pipeline Creation:**

*   **Build the Retrieval Chain:** Assemble the RAG pipeline using Langchain, ChromaDB, and Ollama.

    ```python
    from langchain.chains import RetrievalQA

    # Create the retriever (search in the vector database)
    retriever = db.as_retriever()

    # Build the RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        chain_type="stuff", # other options are 'map_reduce', 'refine'
        retriever=retriever,
        return_source_documents=True # include the retrieved documents in the output
    )
    ```

**7.  Python Script for Godot:**

*   **Encapsulate RAG logic in a Python script:** Create a Python script (e.g., `rag_script.py`) containing all the code from steps 2-6.

**8. Integrate with Godot:**

*   **Godot Setup:**  You'll need to execute the Python script from within Godot.  Here's a basic approach using the `OS.execute()` function in GDScript:

    ```gdscript
    extends Node

    func _ready():
        # Example usage
        var query = "What is the capital of France?"
        var result = ask_chatbot(query)
        print("Chatbot's answer:", result)

    func ask_chatbot(query):
        # Construct the command to execute the Python script
        var python_executable = "python" # or "python3" if needed
        var script_path = "res://rag_script.py" # Assuming rag_script.py is in your resources
        var command = [python_executable, script_path, query]

        # Execute the Python script
        var process = OS.execute(command[0], command.slice(1), true) # shell=true might be needed in some cases

        # Get the output from the Python script
        var output = ""
        while OS.get_process_status(process) != OS.PROCESS_STATE_FINISHED:
            OS.delay_msec(100)  # Wait a bit
        output = OS.get_process_output(process)  #Not Available in Godot 4.2 - see details on last bullets.

        return output # Or parse the output

    ```

    **Important Considerations for Godot Integration:**

    *   **Python Execution:**  `OS.execute()` can be tricky.  Ensure Python is in the system's PATH, or use the full path to the `python.exe` executable.  Experiment with the `shell` argument to see if it's necessary.

    *   **Passing Data to Python:** The example passes the query as command-line arguments. This works, but for more complex data, consider writing the query to a file and having the Python script read from that file, or use JSON serialization.
    *   **Getting Data from Python:** `OS.get_process_output()` does not exist pre-4.3. Check out this example on Godot docs:
        [https://docs.godotengine.org/en/stable/classes/class_os.html#class-os-method-execute](https://docs.godotengine.org/en/stable/classes/class_os.html#class-os-method-execute)

        Another option is to have the python script write to a file (json or text), that you can then read from the Godot side.

    *   **Threading:**  Executing a Python script will block the Godot main thread.  For a smoother experience, use Godot's `Thread` class to run the script in a separate thread.

    *   **Error Handling:** Implement robust error handling to catch exceptions in both the GDScript and the Python code.
    *   **Packaging:**  When you package your game, you'll need to include the Python interpreter and any necessary Python libraries (e.g., as data files or by using a tool like PyInstaller to create a self-contained executable). This significantly increases the size of your game.
    *  **Alternative: Godot Native Extensions**. Consider writing native modules for doing calculations, as this will yield the best performance.

**Example Python (rag_script.py - Modified to Receive Query from Command Line):**

```python
import sys
import chromadb
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Read query from command line
query = sys.argv[1] if len(sys.argv) > 1 else "What is the main topic?"

# 2. Load and Split Documents
loader = TextLoader("data/my_documents.txt")  # Replace with your file
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. Create the Embeddings
model_name = "all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. Store the Embeddings in ChromaDB
client = chromadb.Client()
db = Chroma.from_documents(chunks, embeddings, client=client, collection_name="my_collection")

# 5. Configure Langchain with Ollama
ollama_llm = Ollama(model="llama2")  # Replace 'llama2' if you use another model

# 6. Implement the RAG
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ollama_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. Get the answer and print it
result = qa({"query": query})

print(result["result"]) # only the string answer
#print(result) # print the full dictionary object
```

**Alternative approach for godot integration.**
Since integrating python with Godot is complex and error prone, consider implementing the RAG functionalities in c# using the Godot .NET API.

* First ensure you have the .NET dependencies properly set up with your Godot project.
* Next use the correct implementations of the vector dbs and llm to do the calculations. ChromaDB and Llamaindex both have integrations for C#, however the .NET ecosystem is not as advanced as the Python ecosystem in terms of ML, so you might need to tweak the code from the tutorial to use the existing .NET tooling.

**Important Notes:**

*   **Resource Intensity:** Running language models locally is resource-intensive (CPU, RAM). Optimize your model choice and chunk sizes for performance.
*   **Security:** Running external scripts introduces security risks.  Sanitize inputs carefully.
*   **Packaging:** Consider the extra work and size required to package Python dependencies with your Godot game.
* **Performance:** Vector calculations tend to be slow, be sure to add background threads in Godot to prevent freezing the game.

This detailed breakdown should give you a solid foundation for integrating RAG with Ollama into your Godot project on Windows. Remember to test thoroughly and adjust the steps based on your specific needs and system configuration. 

Thanks for your time. Good luck!!
