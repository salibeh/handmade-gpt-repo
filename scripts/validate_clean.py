#!/usr/bin/env python3
"""Clean-check harness for the Handmade GPT practicum."""
from __future__ import annotations
import argparse, hashlib, json, py_compile, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
 "data.py","common.py","bigram.py","loss-limit.py","uniform-context-model.py",
 "context-model.py","multi_head_model.py","gpt_model.py"
]
REQUIRED={
 "gpt_model.py":["position_embedding","TransformerBlock","LayerNorm","FeedForward","x = x +"],
 "context-model.py":["self.key","self.query","self.value"],
 "multi_head_model.py":["MultiHead"],
}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--execute",action="store_true",help="run every training script after static checks")
 p.add_argument("--timeout",type=int,default=1800)
 p.add_argument("--output",type=Path)
 a=p.parse_args()
 checks=[]
 dataset=ROOT/"input.txt"
 checks.append({"check":"dataset-size","pass":dataset.is_file() and dataset.stat().st_size==1115394,
                "observed":dataset.stat().st_size if dataset.is_file() else None})
 for name in SCRIPTS:
  path=ROOT/"scripts"/name
  ok=path.is_file()
  detail=None
  if ok:
   try: py_compile.compile(str(path),doraise=True); detail="compiled"
   except py_compile.PyCompileError as e: ok=False; detail=str(e)
  checks.append({"check":f"compile:{name}","pass":ok,"detail":detail})
 for name,markers in REQUIRED.items():
  text=(ROOT/"scripts"/name).read_text(encoding="utf-8")
  missing=[marker for marker in markers if marker not in text]
  checks.append({"check":f"architecture:{name}","pass":not missing,"missing":missing})
 try:
  import torch
  runtime={"installed":True,"version":torch.__version__,"cuda":torch.cuda.is_available(),
           "mps":bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available())}
 except Exception as e:
  runtime={"installed":False,"error":str(e)}
 checks.append({"check":"torch-runtime","pass":runtime["installed"],"detail":runtime})
 executions=[]
 if a.execute:
  if not runtime["installed"]: raise SystemExit("--execute requires installed requirements")
  for name in SCRIPTS[2:]:
   run=subprocess.run([sys.executable,str(ROOT/"scripts"/name)],cwd=ROOT,text=True,
                      capture_output=True,timeout=a.timeout)
   executions.append({"script":name,"returncode":run.returncode,
                      "stdout_sha256":hashlib.sha256(run.stdout.encode()).hexdigest(),
                      "stderr_tail":run.stderr[-500:]})
 result={"status":"pass" if all(c["pass"] for c in checks) and
         (not a.execute or all(x["returncode"]==0 for x in executions)) else "fail",
         "scope":"full-execution" if a.execute else "static-plus-runtime",
         "checks":checks,"executions":executions}
 output=json.dumps(result,indent=2)
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(output+"\n",encoding="utf-8")
 print(output)
 raise SystemExit(0 if result["status"]=="pass" else 1)

if __name__=="__main__": main()
