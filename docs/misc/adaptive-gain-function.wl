(* ::Package:: *)

funADC[energy_] := Piecewise[{
{8 * energy + 5000, energy < 1000},
{0.35 * energy + 7000, energy >= 1000 && energy < 20000},
{0.05 * energy + 8000, energy >= 20000}}];


showFunADC:=Show[{
Plot[funADC[x],{x, 0, 48000},
ImageSize->800,
PlotRange->{0, 2^14},
AxesOrigin->{0, 0},
PlotStyle->Thick,
FrameStyle->Directive[Thick, 18],
FrameLabel->{"Energy (keV)", "ADC"},
Frame->True],
Graphics[{
Dashed, Red,
Line[{{1000, 0},{1000, 2^14}}],
Line[{{20000, 0},{20000, 2^14}}]}]
}]
