#!/usr/bin/env python3
"""Download the minimal UTIAS MRCLAM Dataset 1 files used by the real-data demo.

The authoritative dataset description is:
https://asrl.utias.utoronto.ca/datasets/mrclam/
Files are fetched from a public GitHub mirror for reproducible scripted access.
"""
from pathlib import Path
from urllib.request import urlretrieve
import argparse

BASE="https://raw.githubusercontent.com/1988kramer/UTIAS-practice/master/MRCLAM_Dataset1"
COMMON=["Barcodes.dat","Landmark_Groundtruth.dat"]

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="data/external/MRCLAM_Dataset1")
    p.add_argument("--robots",nargs="+",type=int,default=[1]);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    files=COMMON+sum(([f"Robot{r}_Groundtruth.dat",f"Robot{r}_Odometry.dat",f"Robot{r}_Measurement.dat"] for r in a.robots),[])
    for name in files:
        dst=out/name
        if dst.exists(): print("[skip]",dst);continue
        print("[download]",name);urlretrieve(f"{BASE}/{name}",dst)

if __name__=="__main__":main()
