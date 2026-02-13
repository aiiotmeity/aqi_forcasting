@echo off
cd C:\Users\Meity-4\air-quality-forecasting

echo Running update_dataset.py first...
C:\Users\Meity-4\AppData\Local\Programs\Python\Python310\python.exe -u update_dataset.py > update_dataset_run_log.txt 2>&1

echo Running automated_pipeline.py afterwards...
C:\Users\Meity-4\AppData\Local\Programs\Python\Python310\python.exe -u automated_pipeline.py > automated_pipeline_run_log.txt 2>&1

echo Running pipeline_controller.py afterwards...
C:\Users\Meity-4\AppData\Local\Programs\Python\Python310\python.exe -u pipeline_controller.py > pipeline_controller_run_log.txt 2>&1

echo Pipeline execution completed at %date% %time% >> pipeline_run_log.txt