# Deployment runbook

## Local demo
1. Install Docker Engine/Compose or Docker Desktop and start the daemon.
2. Copy `.env.example` to `.env`. Set long random API/WebUI secrets; keep `AGENT_MODE=demo`.
3. Create the local `models/` directory, then run `docker compose config`, `docker compose build`,
   and `docker compose up -d` from the repository root.
4. Check `docker compose ps`, `http://localhost:8000/health`, and `/v1/models` using the API key.
5. Open `http://localhost:3000`, create the initial administrator, select the single agent model,
   and turn Stream Chat Response off in its advanced model parameters.
6. Try documentation, database diagnostics, and production-corruption prompts. Responses must
   disclose mock health and local-only escalation. Check `docker compose logs support-agent` for startup issues.

## Production preparation
1. Train/evaluate the three corrected workflows. Inspect actual outputs, failure cases, baseline
   comparisons, per-class recall, long-context QA and Golden Set. Do not proceed with failed gates.
2. Run real router comparison and `RUN_MODEL_TESTS=1 python -m pytest -q` in the training environment.
3. Transfer the complete three saved model/tokenizer directories to the deployment host's `models/`.
   Never mount a 360M adapter against the 135M base. Include the three model reports and threshold report.
4. Release report paths must refer to the deployment paths, not a Windows training path. Regenerate
   evidence using `python -m scripts.prepare_deployment_release` in the container with writable reports:
   `docker compose run --rm --volume ./reports:/app/reports:rw support-agent python -m scripts.prepare_deployment_release`.
   This verifies source artifact hashes from the original release before remapping paths. Create the
   original release with `python -m src.evaluation.release` before transferring files.
5. Set `AGENT_MODE=production`; keep `MODEL_A_PATH=/app/models/intent_classifier`,
   `MODEL_B_PATH=/app/models/qa_model`, `MODEL_C_ADAPTER=/app/models/support_adapter`.
6. Pin `OPEN_WEBUI_IMAGE` to a tested release tag/digest. Rebuild/restart, verify health is 200 and
   `models_verified=true`, and exercise all four graph routes. Keep the API inaccessible publicly
   until these checks and an independent data review pass.

## Dokploy
1. Push the reviewed implementation branch to your Git provider, then sign in to your Dokploy server.
2. Create a project and a Docker Compose service linked to that repository/branch. Use the root
   `docker-compose.yml` and build context `.`. No server credentials are stored in this repository.
3. Add environment variables in Dokploy: mode, API/WebUI secrets, model paths, and optional Langfuse
   keys/base URL and HF_TOKEN for private downloads. Keep the SQLite/model-cache/WebUI named volumes.
4. Provision the model/report bind mounts in the service's deployment directory (or adapt them to
   host paths you explicitly provision). Complete the production preparation above on that host.
5. Add an HTTPS domain to Open WebUI's container port 8080. The Compose localhost mappings support
   local smoke checks; Dokploy's reverse proxy should connect to the container network/port. Keep
   support-agent internal, or add a separate protected HTTPS domain to port 8000 if external API access is required.
6. Deploy and inspect build/service health. Confirm WebUI uses only `http://support-agent:8000/v1`
   and the configured agent key. Create/administer WebUI users and disable streaming for this model.
7. Test QA, tools, support and escalation through WebUI. With Langfuse enabled, confirm one trace
   contains routing, model/tool spans and redacted input/output. Save the trace URL and demo URL in
   your submission without including secrets.
8. Back up the persistent SQLite/WebUI volumes and record image/model hashes. Roll back to an
   evaluated release rather than editing quality reports. A failed gate means use an explicitly labelled
   demo or postpone deployment; it does not justify bypassing production readiness.

## Required external access
- Hugging Face: no credentials for verified public models; token only for private artifacts or publishing.
- Langfuse: project public/secret keys and a reachable instance for a real trace URL.
- Dokploy: server/project access, configured host storage, domain and TLS for actual deployment.
- Docker: a running engine is necessary for build/config/startup validation.

No live deployment or trace screenshot is claimed unless recorded in the implementation status report.

## Static validation versus deployment
The continuation report distinguishes parsed Compose/Dockerfile checks from a real Docker
build. YAML parsing does not validate image availability, container permissions or networking.
Confirm service domains against https://docs.dokploy.com/docs/core/docker-compose/domains
and provider configuration against https://docs.openwebui.com/reference/env-configuration/.
No retraining is needed to run the demo or software tests. Production still needs accepted,
verified artifacts; do not lower the threshold or fabricate release evidence to start it.
