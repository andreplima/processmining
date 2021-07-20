@echo off

@REM usage:   run <template process> <sample size>
@REM example: run E1 10   -- generate 10 samples from model E1 and corresponding representations

set PYTHONHASHSEED=23
@REM set PARAM_MAXCORES=1

python generate.py %1 %2
python represent.py sample
