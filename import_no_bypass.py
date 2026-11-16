import sys, pathlib, traceback
project_root = r'C:\\Users\\DHRUV\\Desktop\\ai implimented etl'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print('sys.path[0]=', sys.path[0])
try:
    import src.etl.schema_intelligence as s
    print('IMPORT SUCCESS', s.__file__)
except Exception as e:
    print('IMPORT FAILED')
    traceback.print_exc()
