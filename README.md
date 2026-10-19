# First-party phone intent model

Score every phone number in your call log **P1-P4** by how likely a real person is to pick up, from your own
first-party call data. Drop in a CSV export from your dialer or CRM, get back a tier, a score and the best local
time to call each number.

| Tier | Means | Do |
|---|---|---|
| **P1** | High intent: the top 10% of your live numbers | Call first |
| **P2** | Moderate: the next 20% | Call when P1 is done |
| **P3** | Low / no intent: voicemail, screeners, never answers | Call last, or drop |
| **P4** | Bad data: dead, disconnected or wrong numbers | Remove or re-source |

## Quick start

```bash
pip install git+https://github.com/callcloudapp/first-party-phone-intent-model
phone-intent score my_calls.csv --out scores.csv
```

`scores.csv`, sorted P1 first:

| number | tier | score | best_local_time | tries | people_reached | dead_chance | state | reason |
|---|---|---|---|---|---|---|---|---|
| +12125550123 | P1 | 23.4 | Tue 10:30 | 4 | 2 | 0.4 | New York | reached a person 2x, last 6d ago |
| +13055550188 | P3 | 4.1 | Thu 16:30 | 5 | 0 | 2.1 | Florida | no person in 5 tries; voicemail / no answer so far |
| +14155550199 | P4 | 0.2 | Mon 9:30 | 2 | 0 | 100.0 | California | came back as a bad number |

## Your CSV

One row per call. Columns are found by name, case-insensitive:

| Needed | Accepted names |
|---|---|
| the number | `phone`, `phone_number`, `number`, `to`, `dialed`, `contact_phone` (E.164 best; 10 digits = +1) |
| when | `called_at`, `date`, `timestamp`, `created_at`, `start_time`, `time` (ISO 8601, with a UTC offset if you can) |
| what happened | `outcome`, `disposition`, `result`, `status`, `call_result` (free text) |
| *optional* duration | `duration`, `duration_s`, `talk_time`, `seconds` (a long call with no outcome counts as a person) |
| *optional* caller ID | `caller_id`, `from`, `from_number` |
| *optional* carrier lookup | `carrier_family` / `carrier` (verizon, att, tmobile, landline, voip...), `lrn_line_type` / `line_type` (wireless, landline, voip), `ported` (1/0) |

Carrier columns are worth adding. An LRN lookup costs about $0.0001 a number from any LRN vendor. With them, the
tool switches to a second starter model that ranks clearly better (see below).

Outcomes are read as free text. "Connected", "Spoke with", "Not interested", "Meeting booked" count as a person.
"Left voicemail", "No answer", "Busy", "Screener" count as no person. "Wrong number", "Disconnected", "Invalid" count as
bad data. Anything else (cancelled, dropped by the dialer) is ignored rather than counted as a miss. The command prints
the outcomes it did not recognise, so you can see what it skipped.

## What it looks at

Only what's in your own log, plus public numbering-plan facts:

- **The number:** country, area code and state, and its local time zone.
- **When you'd call:** the callee's local hour and weekday. Each number is scored at its best weekday and hour, which is the `best_local_time` column.
- **Its history in your data, and when it happened:** tries, people reached, voicemails and no-answers, and dead results. It also weighs time: days since the last person and since the last miss, the last 30 days on their own, an answer rate that counts recent calls more (30-day half-life), and past answers near this hour and on this weekday.
- **Your caller ID** (if the log has it): whether it's local to the callee.

There are two models: P(a real person answers) and P(the number is dead). P4 comes from the second, plus any number
that came back as a bad number and never reached a person since.

## How good is it

The starter models are trained on about 395,000 real outbound dials from CallCloud customers (mostly US B2B lists).
"A person answered" comes from call transcripts where there is one, so a voicemail a rep clicked Connect on doesn't
count as a person. The models contain only aggregate patterns such as area code, state, hour, history and carrier
effects. They have no phone numbers and no customer data.

Forward test: each customer's call log up to a cut date was scored. The test then checked what happened on the
**next 62,080 calls**, on models that never saw anything after the cut:

| | Plain (number + history) | With carrier columns |
|---|---|---|
| Ranking within each list (AUC) | 0.64 | **0.70** |
| People found in the top 20% of calls | 35% | **41%** |
| P1 reached a person | 13.2% (**3.0x** the list average) | 18.6% (**4.2x**) |
| P4 numbers that were dead | 92% | 92% |

Results are best on numbers you've called before. On a number's very first dial, carrier data is what moves the
score (within-list AUC 0.57 plain, 0.65 with carrier columns). Patterns differ between B2C mobile lists and B2B office
lists, so with enough of your own calls, train your own models (below).

## Train on your own data

With enough of your own calls (tens of thousands), a model trained on them beats the starter model for your lists:

```bash
phone-intent train my_calls.csv --out my_models/
phone-intent score my_calls.csv --models my_models/
```

## Privacy

Everything runs on your machine. Nothing is uploaded.

## License

MIT. Built by [CallCloud](https://www.callcloud.app).
