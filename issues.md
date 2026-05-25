# Potential Issues To Review Later

1. **Critical: Auth can be accidentally turned off in production**

   If `OAUTH_CLIENT_ID` is missing, the backend can fall back to a shared `dev` user. That could expose sessions or allow users to act through shared server credentials.

   **Next action:** Make Cloud Run or production startup fail unless OAuth is configured.

2. **High: Auto fine-tune can push models to the wrong Hugging Face namespace**

   If `HF_NAMESPACE` is not set, auto fine-tuned models default to `ligaments-dev/...`. The job may run with a different user's Hugging Face token, which can fail or write to the wrong place.

   **Next action:** Use the authenticated Hugging Face user as the output namespace unless a namespace is explicitly configured.

3. **High: Dataset repo is preserved, but dataset safety profile may be skipped**

   The chat metadata fix keeps the selected `dataset_repo`, but the auto fine-tune flow may still miss the dataset profile and strategy checks. A risky dataset could avoid the intended direct fine-tune block.

   **Next action:** Persist or reload the dataset profile before auto fine-tuning, then add a regression test.

4. **High: New sessions can race with chat streaming startup**

   The session API can return `ready=true` before all background session pieces are fully initialized. If chat starts immediately, streaming internals such as the broadcaster may not be ready yet.

   **Next action:** Initialize streaming state before returning ready, or make chat wait/retry until the session is fully ready.

5. **Medium: Cloud Run session recovery is local-instance only**

   Session persistence writes to the app filesystem by default. On Cloud Run, that storage is temporary and not shared across instances, so restarts or scaling can lose sessions.

   **Next action:** Use shared durable storage in production, or make it clear that recovery is limited on Cloud Run.

6. **Medium: Claude quota resets on backend restart**

   Daily Claude usage tracking is stored in memory. A restart or another Cloud Run instance can reset usage, weakening cost controls.

   **Next action:** Move quota counters to durable shared storage if quota enforcement protects real billing.

7. **Medium: Session transcript upload may include sensitive user data**

   Session saving can upload scrubbed messages and events to a Hugging Face dataset when an upload token is configured. Scrubbing helps, but it is not a complete privacy guarantee.

   **Next action:** Make transcript upload opt-in for production and document exactly what data is stored.
