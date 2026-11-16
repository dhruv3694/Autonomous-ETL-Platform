import sys, pathlib, importlib.util, traceback

print('--- Diagnostic from project root ---')
print('cwd:', pathlib.Path.cwd())
# Ensure project root is on sys.path
project_root = pathlib.Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print('sys.path first 5:', sys.path[:5])

def find_and_print(name):
    spec = importlib.util.find_spec(name)
    print(f'find_spec {name}:', spec)
    return spec

find_and_print('src')
find_and_print('src.etl')
find_and_print('src.etl.schema_intelligence')

try:
    import src.etl.schema_intelligence as si
    print('Import succeeded, file =', si.__file__)
except Exception:
    print('Import failed:')
    traceback.print_exc()

print('--- End Diagnostic ---')
