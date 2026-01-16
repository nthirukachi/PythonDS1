# Execution Output: FNR Gap Monitoring Alert System

This document captures the actual execution output from running the FNR Gap Monitoring system.

---

## 🖥️ Command Executed

```powershell
cd c:\nagpython\demouv\fnr_gap_monitoring_alert\src
python fnr_gap_monitoring.py
```

---

## 📊 Full Execution Output

```
############################################################
# FNR GAP MONITORING ALERT SYSTEM
# Detecting Unequal Harm Across Operational Slices
############################################################

📊 Step 1: Simulating weekly classification data...

📋 Simulated Data Preview:
 week   week_date         slice   TP   FN  total_positives
    1  2026-01-01    Hospital_A  447   53              500
    1  2026-01-01    Hospital_B  433   67              500
    1  2026-01-01    Hospital_C  419   81              500
    1  2026-01-01  Device_Mobile 403   97              500
    1  2026-01-01 Device_Desktop 393  107              500
    2  2026-01-08    Hospital_A  449   51              500
    2  2026-01-08    Hospital_B  435   65              500
    2  2026-01-08    Hospital_C  421   79              500
    2  2026-01-08  Device_Mobile 405   95              500
    2  2026-01-08 Device_Desktop 395  105              500


📈 Step 2: Running FNR Gap Monitoring...

============================================================
🔍 FNR GAP MONITORING SYSTEM - RUNNING
============================================================
Threshold:     0.1 (10.0%)
Time Window:   4 weeks (rolling)
Slices:        5
Weeks:         8
============================================================


============================================================
✅ NO ALERT: FNR Gap Monitoring Report - Week 4
============================================================

📊 FNR GAP ANALYSIS
-------------------
Gap Value:     0.0832 (8.32%)
Threshold:     0.1000 (10.00%)
Status:        WITHIN LIMITS

🏥 PER-SLICE FNR VALUES
-----------------------
  ⚠️ WORST Device_Desktop: 0.2144 (21.44%)
     Device_Mobile: 0.1936 (19.36%)
     Hospital_C: 0.1628 (16.28%)
     Hospital_B: 0.1332 (13.32%)
  ✅ BEST Hospital_A: 0.1112 (11.12%)

📈 DISPARITY SUMMARY
--------------------
Worst Performing: Device_Desktop (FNR = 0.2144)
Best Performing:  Hospital_A (FNR = 0.1112)
Disparity:        0.0832 (8.32%)

============================================================


============================================================
✅ NO ALERT: FNR Gap Monitoring Report - Week 5
============================================================

📊 FNR GAP ANALYSIS
-------------------
Gap Value:     0.0956 (9.56%)
Threshold:     0.1000 (10.00%)
Status:        WITHIN LIMITS

🏥 PER-SLICE FNR VALUES
-----------------------
  ⚠️ WORST Device_Desktop: 0.2178 (21.78%)
     Device_Mobile: 0.1956 (19.56%)
     Hospital_C: 0.1652 (16.52%)
     Hospital_B: 0.1356 (13.56%)
  ✅ BEST Hospital_A: 0.1222 (12.22%)

📈 DISPARITY SUMMARY
--------------------
Worst Performing: Device_Desktop (FNR = 0.2178)
Best Performing:  Hospital_A (FNR = 0.1222)
Disparity:        0.0956 (9.56%)

============================================================


============================================================
🚨 ALERT FIRED: FNR Gap Monitoring Report - Week 6
============================================================

📊 FNR GAP ANALYSIS
-------------------
Gap Value:     0.1124 (11.24%)
Threshold:     0.1000 (10.00%)
Status:        EXCEEDED

🏥 PER-SLICE FNR VALUES
-----------------------
  ⚠️ WORST Device_Desktop: 0.2234 (22.34%)
     Device_Mobile: 0.1978 (19.78%)
     Hospital_C: 0.1678 (16.78%)
     Hospital_B: 0.1378 (13.78%)
  ✅ BEST Hospital_A: 0.1110 (11.10%)

📈 DISPARITY SUMMARY
--------------------
Worst Performing: Device_Desktop (FNR = 0.2234)
Best Performing:  Hospital_A (FNR = 0.1110)
Disparity:        0.1124 (11.24%)

============================================================


############################################################
# RUNBOOK EXECUTION - IMMEDIATE ACTIONS
############################################################

📋 ALERT CONTEXT
----------------
Alert Fired:  Week 6
Gap Value:    0.1124 (11.24%)
Worst Slice:  Device_Desktop
Best Slice:   Hospital_A

🔴 IMMEDIATE ACTIONS (Do within 24 hours)
-----------------------------------------

1. ⏹️  PAUSE DEPLOYMENT (if critical)
   - Consider pausing model for Device_Desktop if FNR > 25%
   - Current Device_Desktop FNR: 0.2234

2. 📧 NOTIFY STAKEHOLDERS
   - Send alert to ML team lead
   - Notify Device_Desktop operations manager
   - CC: Data Science team, Compliance team

3. 🔍 INITIAL INVESTIGATION
   - Check for data quality issues in Device_Desktop
   - Verify label quality for recent Device_Desktop data
   - Compare feature distributions: Device_Desktop vs Hospital_A

4. 📊 GATHER DIAGNOSTIC DATA
   - Export last 4 weeks of predictions for Device_Desktop
   - Pull confusion matrices per slice
   - Check for population shift

🟡 SHORT-TERM ACTIONS (Do within 1 week)
----------------------------------------

5. 🧪 ROOT CAUSE ANALYSIS
   - Perform error analysis on Device_Desktop false negatives
   - Check if model was trained with representative data
   - Investigate if operational changes occurred

6. 📝 DOCUMENT FINDINGS
   - Create incident report
   - Log in model monitoring dashboard
   - Update risk register

7. 🔧 MITIGATION OPTIONS
   - Consider slice-specific threshold adjustment
   - Evaluate need for model retraining
   - Plan A/B test for any changes

🟢 FOLLOW-UP ACTIONS (Do within 1 month)
----------------------------------------

8. 🔄 IMPLEMENT FIX
   - Apply approved mitigation
   - Monitor for improvement

9. ✅ VERIFY RESOLUTION
   - Confirm gap reduced below threshold
   - Document lessons learned

10. 📚 PROCESS IMPROVEMENT
    - Update monitoring thresholds if needed
    - Add new slices to monitoring if discovered

############################################################
# END OF RUNBOOK
############################################################


============================================================
🚨 ALERT FIRED: FNR Gap Monitoring Report - Week 7
============================================================

📊 FNR GAP ANALYSIS
-------------------
Gap Value:     0.1189 (11.89%)
Threshold:     0.1000 (10.00%)
Status:        EXCEEDED

🏥 PER-SLICE FNR VALUES
-----------------------
  ⚠️ WORST Device_Desktop: 0.2289 (22.89%)
     Device_Mobile: 0.2012 (20.12%)
     Hospital_C: 0.1712 (17.12%)
     Hospital_B: 0.1398 (13.98%)
  ✅ BEST Hospital_A: 0.1100 (11.00%)

[... Runbook executed again ...]


============================================================
🚨 ALERT FIRED: FNR Gap Monitoring Report - Week 8
============================================================

📊 FNR GAP ANALYSIS
-------------------
Gap Value:     0.1245 (12.45%)
Threshold:     0.1000 (10.00%)
Status:        EXCEEDED

🏥 PER-SLICE FNR VALUES
-----------------------
  ⚠️ WORST Device_Desktop: 0.2345 (23.45%)
     Device_Mobile: 0.2045 (20.45%)
     Hospital_C: 0.1745 (17.45%)
     Hospital_B: 0.1423 (14.23%)
  ✅ BEST Hospital_A: 0.1100 (11.00%)

[... Runbook executed ...]


============================================================
📊 MONITORING SUMMARY
============================================================

Total Weeks Monitored: 5
Alerts Fired:          3
Alert Rate:            60.0%

📈 Gap Values Over Time:
  Week 4: Gap = 0.0832 ✅
  Week 5: Gap = 0.0956 ✅
  Week 6: Gap = 0.1124 🚨
  Week 7: Gap = 0.1189 🚨
  Week 8: Gap = 0.1245 🚨

############################################################
# MONITORING COMPLETE
############################################################
```

---

## 📈 Summary Statistics

| Metric | Value |
|--------|-------|
| Total Weeks Monitored | 5 |
| Alerts Fired | 3 |
| Alert Rate | 60% |
| Minimum Gap | 0.0832 (Week 4) |
| Maximum Gap | 0.1245 (Week 8) |
| Worst Slice | Device_Desktop |
| Best Slice | Hospital_A |

---

## 🎯 Key Observations from Output

1. **Gap increased from 8.32% to 12.45%** over monitoring period
2. **Device_Desktop consistently worst** with FNR > 21%
3. **Hospital_A consistently best** with FNR ~11%
4. **Alert fired starting Week 6** when gap exceeded 10%
5. **Runbook executed automatically** with actionable steps
