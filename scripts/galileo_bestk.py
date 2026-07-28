#!/usr/bin/env python3
"""Select k on validation for the reflectance-scaled Galileo, report test metrics.

Matches the SI table convention ("k selected on validation performance"), so the
corrected Galileo row is chosen the same way as every other row rather than by
peeking at the test set.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO/"src"))
from sdg6.knn import _knn_softmax_vote_with_probs

D = REPO/"runs"/"embeddings"/"galileo_scale39_norm"
KS = [5,10,20,50,100,200]; TEMP=0.07
def load(s): 
    d=np.load(D/f"{s}.npz",allow_pickle=True); return d["features"], d
Xtr,dtr=load("train"); Xva,dva=load("val"); Xte,dte=load("test")
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
A=torch.nn.functional.normalize(torch.as_tensor(Xtr,device=dev,dtype=torch.float32),dim=1)
V=torch.nn.functional.normalize(torch.as_tensor(Xva,device=dev,dtype=torch.float32),dim=1)
T=torch.nn.functional.normalize(torch.as_tensor(Xte,device=dev,dtype=torch.float32),dim=1)
for task in ("pw","sw"):
    ytr=dtr[f"{task}_label"].astype(np.int64); yva=dva[f"{task}_label"]; yte=dte[f"{task}_label"]
    y=torch.as_tensor(ytr,device=dev,dtype=torch.long)
    ov=_knn_softmax_vote_with_probs(A,y,V,num_classes=2,k_values=KS,temperature=TEMP)
    ot=_knn_softmax_vote_with_probs(A,y,T,num_classes=2,k_values=KS,temperature=TEMP)
    val_acc={k:accuracy_score(yva,ov[k][0]) for k in KS}
    kstar=max(val_acc,key=val_acc.get)
    p,_,pr=ot[kstar]
    print(f"{task}: k*={kstar} (val_acc={val_acc[kstar]*100:.2f}) | "
          f"test acc={accuracy_score(yte,p)*100:.2f} rec={recall_score(yte,p,average='macro')*100:.2f} "
          f"f1={f1_score(yte,p,average='macro')*100:.2f} auroc={roc_auc_score(yte,pr[:,1])*100:.2f}")
    print("   val_acc by k: "+", ".join(f"{k}:{val_acc[k]*100:.2f}" for k in KS))
