# Labeling Rules — Image Quality Gate

## Classes
- **pass**: Sharp, well-lit, subject fully in frame, GPS + timestamp present
- **blur**: Laplacian variance visually low — soft edges, smeared details
- **exposure**: Clearly too dark OR clearly washed out/overexposed
- **crop_error**: Main subject cut off at any edge, or subject too small (<20% of frame)
- **metadata_fail**: Visual quality is fine BUT GPS or timestamp missing from EXIF

## Edge case rules
- If blur + dark: label whichever is MORE severe
- Slight softness that a client would accept → pass
- Perfect image, missing GPS → metadata_fail (visual quality does not save it)
- Slightly off-centre but subject fully in frame → pass

## What I will NOT label
- Images where I genuinely cannot decide → skip (press q, move to an 'unsure' folder)
- Duplicate images (same scene, nearly identical) → keep only one