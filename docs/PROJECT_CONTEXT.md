# Project Context

## 1. Purpose

This document defines the public scope and contract boundaries for the **Controlled
Mixture-of-Agents System for Smart Construction Safety Monitoring and Reporting**.
It is the reference for code and documentation in this repository.

The broader research project studies a region-grounded safety workflow. This repository
publishes a small shared camera-frame preprocessing boundary alongside the Context and
Rule/Severity slice, deterministic routing, and typed contracts required to keep those
agents safe.

## 2. Current public boundary

### Included

- Deterministic preprocessing of one externally supplied camera frame for visual agents.
- Typed evidence and action contracts.
- `EvidenceSufficiencyGate`.
- `ContextAgent` and its provider-neutral adapter boundary.
- Manifest-bound media resolution for a local Context pilot.
- A loopback-only `llama.cpp` adapter.
- `RuleSeverityAgent` and the initial helmet-zone rule catalog.
- Unit tests for fail-closed routing and validation.
- **Baseline Detector**: YOLO-based detector training, inference, and data preparation (merged into `src/detector_baseline/`).

### Not included

- Person-PPE association and zone-agent implementations (beyond the baseline scripts).
- Camera/RTSP connection, frame sampling, buffering, or end-to-end video ingestion.
- Automatic fulfillment of frame/crop requests.
- Final orchestration, alert policy, reporting, or external notifications.
- Model weights, runtime binaries, private media, or benchmark results.
- Production or field-readiness claims.

## 3. Expected upstream evidence

The parent research system is expected to produce normalized evidence from:

- `person` detections;
- `helmet` or `no_helmet` detections; and
- manually or otherwise explicitly configured zone polygons.

One `CandidateEvent` represents one person in one frame or clip. A person detection is
the target anchor. PPE detections and zone grounding must reference the same target ID.
Missing or mismatched target evidence fails closed.

The current primary research question is limited to the person's zone and helmet state.
Behavior, identity, BIM, motion, and general scene interpretation are not silently added
to the contract.

## 4. Canonical controlled flow

```text
normalized CandidateEvent
-> EvidenceSufficiencyGate
-> READY_FOR_RULE | NEEDS_CONTEXT | UNRESOLVABLE

READY_FOR_RULE
-> RuleSeverityAgent
-> candidate RuleMatch

NEEDS_CONTEXT
-> ContextAgent with gate-authorized actions
-> valid EMIT: merge selected confirmed evidence and rerun the gate
-> REQUEST_*: pending directive for a parent orchestrator
-> ABSTAIN: unresolved result for a parent orchestrator

UNRESOLVABLE
-> no Context call
-> no RuleMatch
-> parent orchestrator decides review or rejection
```

The gate returns a route; it does not make a human-review, rejection, severity, or alert
decision. Those final policies belong to a parent orchestrator.

## 5. Contract ownership

| Capability | Owner | Boundary |
| --- | --- | --- |
| Validate and normalize one externally supplied image frame | `FramePreprocessor` | Does not connect to cameras, enhance evidence, run models, or persist media |
| Validate evidence sufficiency and authorize Context actions | `EvidenceSufficiencyGate` | Does not call a model or decide an alert |
| Propose bounded contextual evidence or acquisition requests | `ContextAgent` | Does not rewrite PPE, zone, identity, violation, or severity |
| Resolve authorized local media | `MediaResolver` | Reads only manifest-authorized files beneath a configured root |
| Call a local multimodal runtime | `LlamaCppContextModelAdapter` | Loopback only, one call, no cloud or heuristic fallback |
| Map ready evidence to a configured rule and candidate severity | `RuleSeverityAgent` | Does not make the final safety decision |
| Review, reject, alert, or report | Parent orchestrator | Outside this initial repository |

If a change gives two components ownership of the same decision, the contract must be
redesigned before that change is accepted.

## 6. Evidence gate

`EvidenceSufficiencyGate` validates:

- required frame, crop, source, target, PPE, and zone fields;
- target and evidence-reference consistency;
- bbox and confidence validity;
- supported zone types;
- PPE conflict or uncertainty;
- recoverability of visual or temporal evidence issues; and
- the bounded Context attempt budget.

It returns one route:

- `READY_FOR_RULE`: normalized evidence is complete and consistent.
- `NEEDS_CONTEXT`: a declared capability can address the ambiguity.
- `UNRESOLVABLE`: evidence is invalid, insufficient, unsupported, unrecoverable, or
  outside the attempt budget.

## 7. Context Agent

`ContextAgent` receives a `ContextRequest` only when:

- the gate returned `NEEDS_CONTEXT`;
- normalized PPE and zone evidence are present; and
- the gate provided a non-empty `allowed_context_actions` list.

The allowed actions are:

- `EMIT_CONTEXT_EVIDENCE`
- `REQUEST_HIGHER_RESOLUTION_CROP`
- `REQUEST_MORE_FRAMES`
- `ABSTAIN`

The agent validates proposal type, action capability, evidence kind and label, subject
and object IDs, frame/crop/zone references, confidence, parameters, attempt budget, and
action/evidence consistency.

Only selected `CONFIRMED` evidence from a valid authorized
`EMIT_CONTEXT_EVIDENCE` result may be merged. The gate must run again after the merge.
`REQUEST_*` remains pending and does not prove that new media was acquired.
`ABSTAIN` fails closed and never fabricates a rule.

### Optional local vision boundary

The local adapter:

- accepts loopback endpoints only;
- uses one request with temperature zero and an explicit timeout;
- narrows the JSON schema to the request contract;
- resolves current media through a trusted manifest;
- records hashes and preprocessing metadata without exposing raw media; and
- converts media, transport, OOM, JSON, or schema failures to `ABSTAIN`.

The adapter cannot change upstream PPE state, zone grounding, target ID, violation,
severity, final review, or reporting.

## 8. Rule/Severity Agent

`RuleSeverityAgent.apply(candidate, gate_result)` runs only when the route is exactly
`READY_FOR_RULE`. It validates normalized target references and supported zone types,
then maps evidence to a rule from `config/rules.json`.

A `RuleMatch` contains:

- `rule_id`
- `violation`
- `severity`
- `confidence`
- `recommended_action`
- `reason`
- `uncertainty`
- `reason_codes`
- `evidence_refs`
- `missing_evidence`

Every emitted rule ID, description, and recommended action must exist in the configured
catalog. Missing mappings and invalid inputs fail closed.

Confirmed Context relations may refine a candidate severity only when all IDs and
frame/crop/zone references remain grounded in the same candidate. The agent does not
discover new objects from raw pixels.

## 9. Supported helmet-zone policy

PPE-required zones:

- `active_work_area`
- `restricted_zone`
- `work_at_height`

PPE-exempt zones:

- `site_office`
- `rest_area`

Unknown or invented zone types do not receive a fallback rule.

## 10. Status language

The following terms are deliberately separate:

- **Implemented:** code and automated contract tests exist.
- **Selected candidate:** a model or runtime has been chosen for a pilot configuration.
- **Validated:** the candidate passed an adjudicated held-out evaluation with recorded
  runtime and safety evidence.
- **Field-ready:** the complete system has deployment evidence in its intended operating
  environment.

This repository demonstrates implemented agent contracts. It does not, by itself,
establish model validation, production readiness, or field readiness.

## 11. Change rules

Changes should preserve these invariants:

1. Keep preprocessed pixels deterministic, RGB, and traceable to the source hash.
2. Keep downstream geometry in source-image coordinates and preserve the letterbox transform.
3. Stay local-first for the optional Context runtime.
4. Fail closed on malformed, invented, missing, or unsupported evidence.
5. Keep `ContextModelAdapter.analyze(ContextRequest) -> ContextProposal` stable unless a
   versioned contract change is approved.
6. Never let Context rewrite upstream PPE or zone evidence.
7. Never let Rule run before `READY_FOR_RULE`.
8. Never create a placeholder `RuleMatch` for request, abstain, review, or rejection.
9. Do not add cloud APIs, new detector classes, behavior, BIM, or field claims without an
   explicit scope decision and corresponding tests.
