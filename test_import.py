import sys, os
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import src.etl.workflow_state as ws
    print('Import success, functions available:', [name for name in dir(ws) if not name.startswith('_')][:5])
except Exception as e:
    import traceback, json
    print('Import failed:', e)
    traceback.print_exc()
