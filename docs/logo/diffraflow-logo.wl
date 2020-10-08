(* ::Package:: *)

genRectangle[cx_,cy_,w_,h_]:=Rectangle[{cx-w/2,cy-h/2},{cx+w/2,cy+h/2}]


logo=Graphics[{Lighter@Blue,
genRectangle[-1.5,-1.75,2,4],
genRectangle[1.75,-1.5,4,2],
genRectangle[1.5,1.75,2,4],
genRectangle[-1.75,1.5,4,2],
Red,Thickness[0.015],
Circle[{0,0},1],
Circle[{0,0},1.5],
Circle[{0,0},2.5],
Circle[{0,0},3]
},
Axes->False,
ImageSize->800,
Background->None]
