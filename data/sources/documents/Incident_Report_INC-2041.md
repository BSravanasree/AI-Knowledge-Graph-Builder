# Incident Report INC-2041: Card-Testing Attack on Travel Merchants

**Severity:** High  **Status:** Resolved  **Owner:** Arjun Mehta, Fraud Analyst

## What happened
Between 01:00 and 04:30 on three consecutive nights, an automated card-testing attack
targeted Travel and Electronics merchants. Most attempts originated from IP ranges in
Dubai and were marked as Foreign Transaction with Location Mismatch.

## Detection
Arjun Mehta detected the attack when the fraud dashboard showed a cluster of
night-time transactions linked to the High Velocity and Low Device Trust risk factors.
Sentinel DeviceID scores for the attacking devices were below 20.

## Response
Velocity rule VR-5 blocked further attempts once a card reached five transactions in
24 hours. Harbor Trust Bank, our issuing partner, reissued 212 affected cards.

## Lessons learned
Night-time transactions combining Foreign Transaction and Low Device Trust should
trigger step-up authentication automatically. Priya Raman approved this change.
