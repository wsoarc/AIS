import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(15,15))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
except ImportError:
    pass
from quiz_app.catalog import BY_ID
from quiz_app.grading import check_in_process
if __name__=='__main__':
    request=json.load(sys.stdin)
    result=check_in_process(BY_ID[request['id']],request['answer'])
    print(json.dumps(result,ensure_ascii=False,allow_nan=True))
