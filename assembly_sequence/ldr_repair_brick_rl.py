#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# repair_ldr.py — id-safe + batch version

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path

STUD = 20
PLATE_H = 8
BRICK_H = 24
EPS = 1e-3

PARTS: Dict[str, Tuple[str,int,int,int,str]] = {
    'BRICK_1x1': ('brick', 1, 1, BRICK_H, '3005.dat'),
    'BRICK_1x2': ('brick', 1, 2, BRICK_H, '3004.dat'),
    'BRICK_1x3': ('brick', 1, 3, BRICK_H, '3622.dat'),
    'BRICK_1x4': ('brick', 1, 4, BRICK_H, '3010.dat'),
    'BRICK_1x6': ('brick', 1, 6, BRICK_H, '3009.dat'),
    'BRICK_1x8': ('brick', 1, 8, BRICK_H, '3008.dat'),
    'BRICK_1x10':('brick', 1,10, BRICK_H, '6111.dat'),
    'BRICK_1x12':('brick', 1,12, BRICK_H, '6112.dat'),
    'BRICK_1x16':('brick', 1,16, BRICK_H, '2465.dat'),
    'PLATE_1x1': ('plate', 1, 1, PLATE_H, '3024.dat'),
    'PLATE_1x2': ('plate', 1, 2, PLATE_H, '3023.dat'),
    'PLATE_1x3': ('plate', 1, 3, PLATE_H, '3623.dat'),
    'PLATE_1x4': ('plate', 1, 4, PLATE_H, '3710.dat'),
    'PLATE_1x6': ('plate', 1, 6, PLATE_H, '3666.dat'),
    'PLATE_1x8': ('plate', 1, 8, PLATE_H, '3460.dat'),
    'PLATE_1x10':('plate', 1,10, PLATE_H, '4477.dat'),
    'PLATE_1x12':('plate', 1,12, PLATE_H, '60479.dat'),
}

ALLOWED_LENGTHS = {1,2,3,4,6,8,10,12,16}
FILENAME_TO_KEY: Dict[str,str] = {v[-1].lower(): k for k, v in PARTS.items()}

@dataclass
class Part:
    idx: int
    kind: str
    sx: int
    sz: int
    h: int
    fn: str
    color: int
    x: int
    y: int
    z: int
    rot90: bool = False
    def studs_bbox(self): 
        x0 = round(self.x/STUD); z0 = round(self.z/STUD)
        if not self.rot90: return x0,x0+self.sx,z0,z0+self.sz
        return x0,x0+self.sz,z0,z0+self.sx
    def layer(self): return round(self.y/PLATE_H)
    def height_in_plates(self): return round(self.h/PLATE_H)

@dataclass
class Model:
    parts: List[Part] = field(default_factory=list)

def parse_ldr_line(line:str):
    toks=line.strip().split()
    if not toks or toks[0]!="1" or len(toks)<15: return None
    try:
        color=int(toks[1]); x,y,z=map(float,toks[2:5])
        m=list(map(float,toks[5:14])); sub=toks[14].lower()
        return color,x,y,z,m,sub
    except: return None

def load_ldr(path:str)->Model:
    parts=[]; idx=0
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            if not line.lstrip().startswith('1 '): continue
            parsed=parse_ldr_line(line)
            if not parsed: continue
            color,x,y,z,m,sub=parsed
            rot90=False
            if (abs(m[0]-1)<EPS and abs(m[4]-1)<EPS and abs(m[8]-1)<EPS):
                rot90=False
            elif (abs(m[0])<EPS and abs(m[2]-1)<EPS and abs(m[6]+1)<EPS and abs(m[4]-1)<EPS):
                rot90=True
            else: continue
            key=FILENAME_TO_KEY.get(sub)
            if not key:
                low=sub.replace('.dat','')
                n=None
                if '1x' in low:
                    tail=''.join(ch for ch in low.split('1x')[-1] if ch.isdigit())
                    if tail:
                        try:n=int(tail[:2])
                        except:pass
                if n and f'BRICK_1x{n}' in PARTS:key=f'BRICK_1x{n}'
                elif n and f'PLATE_1x{n}' in PARTS:key=f'PLATE_1x{n}'
                else:continue
            kind,sx,sz,h,fn=PARTS[key]
            X=int(round(x/STUD)*STUD); Y=int(round(y/PLATE_H)*PLATE_H); Z=int(round(z/STUD)*STUD)
            parts.append(Part(idx,kind,sx,sz,h,fn,color,X,Y,Z,rot90)); idx+=1
    return Model(parts)

def write_ldr(model:Model,out_path:Path):
    out_path.parent.mkdir(parents=True,exist_ok=True)
    lines=["0 Repaired","0 ROTATION CONFIG 0 0"]
    for p in model.parts:
        if not p.rot90:a,b,c,d,e,f,g,h,i=1,0,0,0,1,0,0,0,1
        else:a,b,c,d,e,f,g,h,i=0,0,1,0,1,0,-1,0,0
        lines.append(f"1 {p.color} {p.x} {p.y} {p.z} {a} {b} {c} {d} {e} {f} {g} {h} {i} {p.fn}")
    out_path.write_text("\n".join(lines),encoding='utf-8')

def idmap(model): return {p.idx:p for p in model.parts}

def layer_graph(model):
    parts=model.parts; by_layer=defaultdict(list)
    for p in parts: by_layer[p.layer()].append(p)
    g=defaultdict(set)
    for u in parts:
        top=u.layer()+u.height_in_plates()
        for v in by_layer.get(top,[]):
            ux0,ux1,uz0,uz1=u.studs_bbox()
            vx0,vx1,vz0,vz1=v.studs_bbox()
            if (ux0<vx1 and vx0<ux1) and (uz0<vz1 and vz0<uz1):
                g[u.idx].add(v.idx); g[v.idx].add(u.idx)
    return g

def components(model,graph):
    seen=set(); comps=[]
    for p in model.parts:
        if p.idx in seen: continue
        comp=set(); stack=[p.idx]
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); comp.add(u)
            for v in graph.get(u,set()):
                if v not in seen: stack.append(v)
        comps.append(comp)
    return comps

def is_grounded_component(model,comp):
    id2=idmap(model); return any(abs(id2[i].y)<EPS for i in comp if i in id2)

def split_components(model):
    g=layer_graph(model); comps=components(model,g)
    grounded,isolated=[],[]
    for c in comps:(grounded if is_grounded_component(model,c) else isolated).append(c)
    return grounded,isolated

def stud_rect(p):ly0=p.layer();ly1=ly0+p.height_in_plates();x0,x1,z0,z1=p.studs_bbox();return ly0,ly1,x0,x1,z0,z1
def rect_touching(a,b):
    a0,a1,ax0,ax1,az0,az1=a;b0,b1,bx0,bx1,bz0,bz1=b
    if a0!=b0 or a1!=b1:return False
    if az0==bz0 and az1==bz1 and (ax1==bx0 or bx1==ax0):return True
    if ax0==bx0 and ax1==bx1 and (az1==bz0 or bz1==az0):return True
    return False
def rect_union_if_colinear(a,b):
    a0,a1,ax0,ax1,az0,az1=a;b0,b1,bx0,bx1,bz0,bz1=b
    if a0!=b0 or a1!=b1:return None
    if az0==bz0 and az1==bz1:return a0,a1,min(ax0,bx0),max(ax1,bx1),az0,az1
    if ax0==bx0 and ax1==bx1:return a0,a1,ax0,ax1,min(az0,bz0),max(az1,bz1)
    return None
def aligned_1xN(r):_,_,x0,x1,z0,z1=r;sx=x1-x0;sz=z1-z0;return (sx==1 and sz>=1) or (sz==1 and sx>=1)
def make_part_from_rect(kind,r,color):
    ly0,ly1,x0,x1,z0,z1=r;sx=x1-x0;sz=z1-z0
    rot90=False;L=0
    if sx==1 and sz>=1:rot90=False;L=sz
    elif sz==1 and sx>=1:rot90=True;L=sx
    else:return None
    if L not in ALLOWED_LENGTHS:return None
    key=f"{'BRICK' if kind=='brick' else 'PLATE'}_1x{L}"
    if key not in PARTS:return None
    k,sx1,sz1,h,fn=PARTS[key];x=x0*STUD;y=ly0*PLATE_H;z=z0*STUD
    return Part(-1,k,sx1,sz1,h,fn,color,x,y,z,rot90)

def try_merge(model,isolated):
    if not isolated:return False
    id2=idmap(model);rects={i:stud_rect(p) for i,p in id2.items()}
    for i in set().union(*isolated):
        p=id2.get(i); 
        if not p or p.kind!='brick':continue
        ai=rects[i]
        for j,bj in rects.items():
            if j==i:continue
            if not rect_touching(ai,bj):continue
            u=rect_union_if_colinear(ai,bj)
            if not u or not aligned_1xN(u):continue
            newp=make_part_from_rect('brick',u,p.color)
            if not newp:continue
            keep=[q for q in model.parts if q.idx not in {i,j}]
            newp.idx=max((q.idx for q in keep),default=-1)+1
            model.parts=keep+[newp];return True
    return False

def try_vertical_support(model,grounded,isolated):
    if not isolated:return False
    id2=idmap(model)
    iso=[id2[i] for i in set().union(*isolated) if i in id2]
    if not iso:return False
    iso.sort(key=lambda p:p.y,reverse=True)
    for p in iso:
        x0,x1,z0,z1=p.studs_bbox()
        for sx_stud,sz_stud in [(x0,z0),(x1-1,z0),(x0,z1-1),(x1-1,z1-1)]:
            ly=p.layer()
            if ly==0:continue
            rem=ly;toadd=[]
            while rem>0:
                if rem>=3:
                    r=(ly-3,ly,sx_stud,sx_stud+1,sz_stud,sz_stud+1)
                    newp=make_part_from_rect('brick',r,14)
                    if not newp:break
                    toadd.append(newp);rem-=3;ly-=3
                else:
                    r=(ly-1,ly,sx_stud,sx_stud+1,sz_stud,sz_stud+1)
                    newp=make_part_from_rect('plate',r,14)
                    if not newp:break
                    toadd.append(newp);rem-=1;ly-=1
            if toadd:
                base=max((p.idx for p in model.parts),default=-1)+1
                for k,np in enumerate(toadd):np.idx=base+k;model.parts.append(np)
                return True
    return False

def repair_once(model,order):
    grounded,isolated=split_components(model)
    if not isolated:return False
    for op in order:
        if op=='merge' and try_merge(model,isolated):return True
        if op=='support' and try_vertical_support(model,grounded,isolated):return True
    return False

def run_repair_file(inp,outp,max_steps,order):
    model=load_ldr(inp)
    steps=0
    while steps<max_steps:
        _,iso=split_components(model)
        if not iso:break
        changed=repair_once(model,order)
        if not changed:break
        steps+=1
    write_ldr(model,Path(outp))
    return steps

def run_cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--in',dest='inp',required=True)
    ap.add_argument('--out',dest='outp',required=True)
    ap.add_argument('--max_steps',type=int,default=20)
    ap.add_argument('--order',type=str,default='merge,support')
    a=ap.parse_args()
    inp,oup=Path(a.inp),Path(a.outp)
    if inp.is_file():
        out=oup/(inp.stem+'_repaired.ldr') if oup.is_dir() else oup
        s=run_repair_file(str(inp),str(out),a.max_steps,a.order.split(','))
        print(f"[OK] {inp.name} -> {out.name} | steps={s}")
    elif inp.is_dir():
        oup.mkdir(parents=True,exist_ok=True)
        fs=sorted(inp.glob('*.ldr'))
        for i,f in enumerate(fs,1):
            out=oup/f"{f.stem}_repaired.ldr"
            try:
                s=run_repair_file(str(f),str(out),a.max_steps,a.order.split(','))
                print(f"[{i}/{len(fs)}] {f.name} -> {out.name} | steps={s}")
            except Exception as e:
                print(f"[{i}/{len(fs)}] {f.name} -> ERROR: {e}")
    else:
        raise FileNotFoundError(inp)

if __name__=='__main__':
    run_cli()
