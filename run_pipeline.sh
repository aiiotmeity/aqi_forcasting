#!/bin/bash

echo "Step 1"
python update_dataset.py

echo "Step 2"
python automated_pipeline.py

echo "Step 3"
python pipeline_controller.py

echo "Pipeline completed"
