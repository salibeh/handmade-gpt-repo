#!/usr/bin/env python3
"""Clean-check and execution harness for the Handmade GPT practicum."""
from __future__ import annotations
import argparse, json, os, py_compile, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
 "data.py","common.py","bigram.py","loss-limit.py","uniform-context-model.py",
 "context-model.py","multi_head_model.py","gpt_model.py"
]
EXECUTABLES=SCRIPTS[2:]
REQUIRED={
 "gpt_model.py":["position_embedding","TransformerBlock","LayerNorm","FeedForward","x = x +"],
 "context-model.py":["self.key","self.query","self.value"],
 "multi_head_model.py":["MultiHead"],
}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--execute",action="store_true",help="execute entropy and every training stage")
 p.add_argument("--mode",choices=["learning","evidence"],default="learning",
                help="learning: 200 steps/20 eval batches; evidence: 10000/200")
 p.add_argument("--train-steps",type=int,help="override mode training steps")
 p.add_argument("--eval-iters",type=int,help="override mode evaluation batches")
 p.add_argument("--timeout",type=int,default=1800,help="seconds allowed per script")
 p.add_argument("--output",type=Path)
 a=p.parse_args()
 defaults={"learning":(200,20),"evidence":(10000,200)}
 train_steps=a.train_steps or defaults[a.mode][0]
 eval_iters=a.eval_iters or defaults[a.mode][1]
 checks=[]
 dataset=ROOT/"input.txt"
 checks.append({"check":"dataset-size","pass":dataset.is_file() and dataset.stat().st_size==1115394,
                "observed":dataset.stat().st_size if dataset.is_file() else None})
 for name in SCRIPTS:
  path=ROOT/"scripts"/name
  ok=path.is_file(); detail=None
  if ok:
   try: py_compile.compile(str(path),doraise=True); detail="compiled"
   except py_compile.PyCompileError as e: ok=False; detail=str(e)
  checks.append({"check":f"compile:{name}","pass":ok,"detail":detail})
 for name,markers in REQUIRED.items():
  source=(ROOT/"scripts"/name).read_text(encoding="utf-8")
  missing=[marker for marker in markers if marker not in source]
  checks.append({"check":f"architecture:{name}","pass":not missing,"missing":missing})
 try:
  import torch
  if torch.cuda.is_available(): selected="cuda"
  elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): selected="mps"
  else: selected="cpu"
  runtime={"installed":True,"version":torch.__version__,"cuda":torch.cuda.is_available(),
           "mps":bool(getattr(torch.backends,"mps",None) and torch.backends.mps.is_available()),
           "selected":selected}
 except Exception as e:
  runtime={"installed":False,"error":str(e)}
 checks.append({"check":"torch-runtime","pass":runtime["installed"],"detail":runtime})
 executions=[]
 if a.execute:
  if not runtime["installed"]: raise SystemExit("--execute requires installed requirements")
  env=os.environ.copy()
  env["HANDMADE_GPT_TRAIN_STEPS"]=str(train_steps)
  env["HANDMADE_GPT_EVAL_ITERS"]=str(eval_iters)
  print(f"execution mode={a.mode} device={runtime['selected']} train_steps={train_steps} eval_iters={eval_iters}",flush=True)
  for index,name in enumerate(EXECUTABLES,1):
   print(f"\n[{index}/{len(EXECUTABLES)}] starting {name}",flush=True)
   started=time.monotonic()
   try:
    run=subprocess.run([sys.executable,str(ROOT/"scripts"/name)],cwd=ROOT,env=env,
                       timeout=a.timeout)
    elapsed=round(time.monotonic()-started,3)
    executions.append({"script":name,"returncode":run.returncode,"elapsed_seconds":elapsed})
    print(f"[{index}/{len(EXECUTABLES)}] finished {name}: returncode={run.returncode} elapsed={elapsed}s",flush=True)
    if run.returncode!=0: break
   except subprocess.TimeoutExpired:
    elapsed=round(time.monotonic()-started,3)
    executions.append({"script":name,"returncode":124,"elapsed_seconds":elapsed,"error":"timeout"})
    print(f"[{index}/{len(EXECUTABLES)}] timed out {name} after {elapsed}s",flush=True)
    break
 static_pass=all(c["pass"] for c in checks)
 execution_pass=not a.execute or (len(executions)==len(EXECUTABLES) and all(x["returncode"]==0 for x in executions))
 result={"status":"pass" if static_pass and execution_pass else "fail",
         "scope":f"{a.mode}-execution" if a.execute else "static-plus-runtime",
         "configuration":{"train_steps":train_steps,"eval_iters":eval_iters,
                          "timeout_per_script":a.timeout},
         "checks":checks,"executions":executions}
 output=json.dumps(result,indent=2)
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True)
  a.output.write_text(output+"\n",encoding="utf-8")
 print("\n"+output,flush=True)
 raise SystemExit(0 if result["status"]=="pass" else 1)

if __name__=="__main__": main()
