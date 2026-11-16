import sys, pathlib, importlib.util, traceback
print('--- Diagnostic 2 ---')
print('cwd:', pathlib.Path.cwd())
print('sys.path first entries:', sys.path[:5])
print('find_spec src:', importlib.util.find_spec('src'))
print('find_spec src.etl:', importlib.util.find_spec('src.etl'))
print('find_spec src.etl.schema_intelligence:', importlib.util.find_spec('src.etl.schema_intelligence'))
try:
    import src.etl.schema_intelligence as si
    print('Import success, file:', si.__file__)
except Exception as e:
    print('Import error:')
    traceback.print_exc()
