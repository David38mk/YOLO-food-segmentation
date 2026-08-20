import os

bad = 0
for root,_,files in os.walk("../yolo_final/labels"):
    for f in files:
        for ln in open(os.path.join(root,f)):
            p = ln.split()
            if len(p)!=5 or p[0]!="0":
                bad+=1
print("Bad labels:", bad)
