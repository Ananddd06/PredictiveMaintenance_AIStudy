import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open("/Users/anand/Desktop/FAI/Notebooks/model_training.ipynb", "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
script, _ = python_exporter.from_notebook_node(notebook)

# Save as .py
with open("your_notebook.py", "w", encoding="utf-8") as f:
    f.write(script)

print("Notebook converted to Python script successfully!")
