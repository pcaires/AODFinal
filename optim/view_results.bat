@echo off
:a
echo Complexity level select:
echo 1- DV: aoa, span
echo 2- DV: aoa, spar thickness
echo 3- DV: aoa, chord scaling, twist, sweep
echo 4- DV: all
echo 5- Exit
echo ----------

set /p SEL="Selection: "

if %SEL% == 5 (exit)

plot_wing CRJ700optim_%SEL%.db 1.5
goto a
