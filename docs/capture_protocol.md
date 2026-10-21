# Controlled photo capture protocol

This protocol is for a new, controlled dataset collection intended to reduce capture confounds and to make the classifier more defensible.

## Goal
Collect a dataset where real and fake saffron samples are photographed under the same conditions unless a deliberate variation is being studied. The objective is to reduce lighting, flash, camera, ISO, and crop-driven shortcuts.

## Required setup
- Use at least 3 phones/models in the same session, ideally including one device with the previously confounded camera behavior.
- For each phone, use the same lens and fixed camera settings for both classes.
  Prefer manual exposure in a fixed light box; otherwise record the automatic
  exposure metadata and treat it as a capture variable.
- Keep the same background, distance, framing, white balance, and illumination
  for every sample in a phone/session block.
- Photograph real and fake saffron interleaved in a randomized order within
  each phone/session rather than running one class at a time.
- Turn flash off for all captures in the primary control set.
- Keep ISO and exposure time fixed or logged in a table for each capture.
- Avoid manual cropping and class-dependent processing after capture. If
  cropping is necessary for an analysis, use the same predeclared crop rule
  for every image and preserve the original files.

## Capture steps
1. Prepare one background setup and one fixed camera mount.
2. Use the same lighting conditions for all images in a block.
3. Record: date and time, phone model, physical sample ID, label, batch/source,
   session, ISO, exposure/shutter, flash state, white balance, lens, and
   background ID in a capture log.
4. Randomize and interleave real and fake samples within each phone/session to
   reduce time-of-session and lighting drift.
5. Shoot at least 30 independent samples per class per phone; this is the minimum for stable group folds and a defensible controlled evaluation.
6. Save raw camera output or minimally processed output in a reproducible archive. Do not re-save files in a way that changes the original measurement distribution.
7. For any multi-shot sample, make sure the sample is clearly linked by a single physical sample ID so group-based splitting can work.

## Data hygiene
- Do not manually crop or resize images after capture.
- Do not use different backgrounds or distances across classes.
- Do not mix flash-on and flash-off captures in the same headline analysis unless the flash status is explicitly separated.
- Keep a per-session log with timestamps and image names.

## Recommended analysis split
- Train/validation/test should always be split at the physical-sample group level, not the image level.
- Report camera-level and flash-status breakdowns separately.
- A model that only works on a flash-heavy camera or ISO-confounded session should not be described as a general saffron detector.
