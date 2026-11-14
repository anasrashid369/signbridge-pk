# SignBridge PK: User Test Plan

Owner: Aliza. This plan is for a small test of the local web application with a few people.
Its purpose is to see what people try and what confuses them. It is not a measurement of
accuracy, and with 3 to 5 people it cannot give percentages. Write down only what happens.

## 1. Setup

- App: the local web application on Anas's laptop (or by screen share), at `127.0.0.1:7860`,
  with the CNN and the landmark SVM and the landmark overlay.
- Participants: 3 to 5 people, about 10 minutes each. They do not need to know sign language.
- Reference: give each participant a printed chart of the ASL alphabet, because the models
  were trained on ASL letters, not PSL. Say so to every participant before they start.
- The observer (Aliza) watches, takes notes and does not help unless the participant is
  stuck for more than one minute. Every help given is written down.
- Keep the first participant's session as a rehearsal if something in this plan does not work.
  Say in the report if a session was a rehearsal.

## 2. Consent line (read aloud before starting)

"We are testing a student project that guesses a letter from a photo of a hand. We will write
down what you do and say, without your name. You can stop at any time and you can skip any
task. [Before the first session, confirm with Anas what the app does with uploaded and webcam
images, and say it here exactly. Do not promise that images are not kept unless that is true.]
Do you agree to take part?"

Record only a yes or no and the participant number. Do not write names.

## 3. Tasks (same order for everyone)

1. **Upload.** Upload a photo of a hand showing a letter from the chart.
2. **Webcam.** Show a letter from the chart to the webcam.
3. **No hand.** Upload a photo with no hand in it (for example a desk or a wall).
4. **A hard letter.** Show M or N (closed-fist letters), and then the same letter in the other
   model mode.
5. **Notice.** Ask, without pointing: "Is there anything on the page that tells you what the
   model was trained on?"

## 4. What to record for each participant

| Field | What to write |
|---|---|
| Participant | P1, P2, ... (no names) |
| Date, mode | Date; which model (CNN or landmark SVM); upload or webcam |
| Task | 1 to 5 |
| What they did | What they actually did, step by step, including mistakes |
| What the app showed | Top letter and the three shown letters, or the no-hand message |
| Match with the chart | Correct, wrong or not applicable (no hand). Decided from the chart, not from memory |
| Confusion | Where they hesitated or asked a question, in their own words |
| Help given | What the observer said or did, and when |
| Noticed the ASL note | Yes or no, and what they said for task 5 |
| Other comments | Quotes, in the participant's words |

## 5. Rules for reporting

- Report what happened, per participant. Do not average, and do not give percentages.
- Say how many people took part and who they were in general terms (for example "four
  classmates, none of whom knew ASL").
- Quote short comments in the participant's words. Do not rewrite them to sound better.
- Report problems as well as successes, and say how many participants hit each problem.
- State the limits: few people, not PSL users, a hand held by a novice, one laptop and camera.
- Never write a result that was not observed. If a task was skipped, write "skipped".
