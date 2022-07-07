@echo off
echo Complexity level select:
echo 1- DV: aoa, span
echo 2- DV: aoa, spar thickness
echo 3- DV: aoa, chord scaling, twist, sweep
echo 4- DV: all
echo ----------
set /p SEL="Selection: "

plot_wing CRJ700optim_%SEL%.db 1.5
