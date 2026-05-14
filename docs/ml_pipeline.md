# Final ML Pipeline

Raw TX/RX IQ  
→ Metadata generation  
→ Preprocessing  
→ Windowing  
→ TX-RX calibration  
→ RX pair-wise AoA/RSSI/SCM feature extraction  
→ All four-pair feature merging  
→ Final dataset creation  
→ Repetition-wise train/validation/test split  
→ Extra Trees AoA model  
→ XGBoost RSSI distance model  
→ Hybrid AoA + distance 2D localization  
→ Feedback correction loop  
→ Final transmitter location  
→ Performance evaluation
