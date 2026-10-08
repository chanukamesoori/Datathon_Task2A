"""Start a fresh workspace kernel and run every cell, failing on errors."""
from pathlib import Path
import os,json,sys,shutil
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'.jupyter/kernels/task2a'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Task2A workspace Python','language':'python'}))
os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter/runtime')
os.environ['IPYTHONDIR']=str(ROOT/'.ipython')
os.environ['LOKY_MAX_CPU_COUNT']='4'
path=ROOT/'notebooks/Task2A_Complete_ML_Forecasting.ipynb'
nb=nbformat.read(path,as_version=4)
def progress(cell_index,**kwargs):print('Executing cell',cell_index,flush=True)
client=NotebookClient(nb,timeout=1200,kernel_name='task2a',resources={'metadata':{'path':str(ROOT)}},allow_errors=False,on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
assert all(o.output_type!='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
shutil.copyfile(path,ROOT/'notebooks/Task2A_Forecasting.ipynb')
print('PASS: fresh-kernel execution; all code cells executed; no error outputs; both notebook names saved.')
