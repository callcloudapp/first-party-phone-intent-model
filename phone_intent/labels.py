"""What happened on a dial, as the model learns it.

Every dial gets one of:
  PERSON     a live human answered (the thing we predict)
  NO_PERSON  voicemail, a call screener, a phone menu, no answer, busy, or the rep judged "not a person"
  DEAD       the number does not work (unallocated, bad number); also a NO_PERSON for the score
  None       not informative: the dialer itself cut the line before anyone could know (another line connected
             first, the rep was busy, the campaign stopped). Counting these as "no answer" would teach the model
             that numbers dialed in parallel never answer.

label_callcloud is for CallCloud's dial log. Any other export is mapped by free text in phone_intent/score.py
(normalize_outcome), which `phone-intent score` and `phone-intent train` use.
"""
from __future__ import annotations

PERSON, NO_PERSON, DEAD = "person", "no_person", "dead"

# Our side ended the call before an outcome existed.
_OUR_CUT = {
    "requeued", "rep_not_available", "campaign_stopped", "cancelled_due_to_connected_call", "stale_reaped",
    "browser_reconnect", "unattended_timeout", "queue_abandoned", "queue_expired_reaped", "hung_up_unconfirmed",
    "normal_clearing", "ended_early",
}
_DEAD = {"unallocated_number", "bad_number"}
_NO_PERSON_DISPOSITIONS = {
    "voicemail_skipped", "voicemail_left", "voicemail", "ai_screener", "automated_system", "no_one_there", "busy",
    "no_answer", "skipped",
}
_NO_PERSON_REASONS = {"no_answer_timeout", "no-answer", "user_busy", "rep_skipped", "rep_marked_voicemail",
                      "queue_machine_ai", "queue_expired_machine"}
_AI_NOT_PERSON = {"voicemail", "no_one_there", "bad_number"}


_TX_NOT_PERSON = {"voicemail", "screener", "ivr", "silence"}
_REP_SAID_VOICEMAIL = {"rep_marked_voicemail"}


def label_callcloud(disposition: str | None, failure_reason: str | None, ai: str | None,
                    line_s: float | None, transcript: str | None = None) -> str | None:
    """`transcript`: the voicemail detector's nightly verdict on the call's recording (person | voicemail | screener |
    ivr | silence | unclear), when there is one. It overrides the dialer's outcome, two ways:
      person     a human picked up, even on a call our dialer then cut (rep busy): newly informative. Except when the
                 rep pressed Voicemail: robotic number read-outs fool the labeller, and the rep heard the call.
      not person voicemail / screener / phone menu / silence, even when the rep clicked Connect (to leave a message).
    `unclear` or none = the dialer's outcome below."""
    d = (disposition or "").strip()
    fr = (failure_reason or "").strip()
    a = (ai or "").strip()
    tx = transcript.strip() if isinstance(transcript, str) else ""
    if d in _DEAD or fr in _DEAD:
        return DEAD
    if tx == "person" and fr not in _REP_SAID_VOICEMAIL:
        return PERSON
    if tx in _TX_NOT_PERSON:
        return NO_PERSON
    if d == "connected":
        if a in _AI_NOT_PERSON:
            return DEAD if a == "bad_number" else NO_PERSON
        return PERSON
    if d in _DEAD or fr in _DEAD:
        return DEAD
    if d == "skipped" and fr in _OUR_CUT:
        return None  # answered, then dropped by us (rep busy): unknown who it was
    if d in _NO_PERSON_DISPOSITIONS:
        return NO_PERSON
    if not d and fr == "completed":
        # finished normally after being on the rep's line: a conversation if it lasted
        return PERSON if (line_s or 0) >= 10 else None
    if fr in _OUR_CUT:
        return None
    if fr in _NO_PERSON_REASONS:
        return NO_PERSON
    return None
