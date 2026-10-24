# SignBridge PK: Data Collection Protocol and Consent Note

This protocol covers the PSL (Pakistan Sign Language) photos collected by our
team. The Kaggle ASL baseline is separate and is not covered here.

## 1. Scope
- Static letter signs only. No motion, no words, no sentences.
- Letters to collect: [not confirmed yet]
- Label names: each letter gets a short ASCII label with no underscores or
  spaces (for example, a romanized name). Labels are listed here and in
  `data/processed/label_map.json`: [not decided yet]
- Sign shapes are checked against: [not verified yet. No PSL dictionary, association or PSL user has been consulted so far]

## 2. Volunteers
- Target: 5 to 10 volunteers.
- Each volunteer gets an anonymous ID: p01, p02, p03, ...
- No names or contact details are stored with the photos or in the repo.
- A private sheet kept outside the repo links each ID to a first name or
  contact, only so a volunteer can later ask for their photos to be deleted.
  This sheet is never committed or shared with assessors.

## 3. Consent note (read to each volunteer before shooting)
"Your hand photos will be used only for a university project that builds a
Pakistan Sign Language alphabet recognizer. Please keep your face out of the
frame. Photos are stored under an anonymous ID, never your name. They are not
uploaded to GitHub and are not shared outside the project team and course
assessors. You may stop at any time and ask us to delete your photos.

We would also like to ask which broad skin tone group you belong to (light,
medium or dark). We ask so that we can check whether the recognizer works
equally well for different people. This question is optional and you can skip
it. Your answer is kept only in our private sheet, never with the photos or on
GitHub, and we never guess it from your photos."

Consent (yes or no per volunteer ID) is recorded in the private sheet.

## 4. What to capture
- Every letter in section 1, for every volunteer.
- Right hand by default. If a volunteer signs left-handed, note it in the
  private sheet.
- Hand positions: [to be agreed by the team before collection starts]
- Lighting: normal, dim, bright.
- Background: plain, cluttered.
- Minimum per letter per volunteer: 10 images spread across these variations
  (proposed; confirm as a team).
- Keep the whole hand visible. Avoid heavy motion blur.
- Optional skin tone group: see section 7. It is self-reported, never guessed
  from a photo, and kept in the private sheet only.

## 5. Storage and naming
- Save into the PSL folder set in `configs/data.yaml` (`data/raw/psl`).
  Never commit it.
- File name: `{person_id}_{label}_{lighting}_{background}_{index}.jpg`
  Example: `p03_alif_dim_plain_004.jpg`
- Use only these values: lighting is one of `normal`, `dim`, `bright`;
  background is one of `plain`, `cluttered`. No underscores inside a value.
- The manifest builder reads person_id, label, lighting and background from
  the file name, so do not change this pattern.

## 6. After collection
- Run the cleaning scripts (`data_scripts/clean/`) before building the manifest.
- If a volunteer withdraws consent, delete all files starting with their
  person_id.

## 7. Skin tone data and the bias analysis
- The skin tone group is optional. The volunteer chooses it themselves from a
  coarse scale (light, medium, dark, or prefer not to say; proposed, confirm as
  a team) after hearing the consent note, and may skip it. We never guess it
  from a photo.
- It is kept only in the private sheet outside the repo. It is not a column of
  the committed `data/processed/manifest.csv`.
- For the bias analysis it is joined to the results on `person_id` on a local
  machine. Only aggregate results are committed.
- A group with fewer than 3 volunteers is never reported, because a group of 1
  or 2 people could be identified.
- With only a few volunteers, the bias check is descriptive: it shows what we
  observed, not a statistical test, and the report says so.
