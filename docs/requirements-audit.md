# Google QUEST requirements audit

## Presentation update — standard 1.1

The current primary display pairs **Public LB** and **Private LB** scores with
**Public Top %**. Rank/team counts and ties remain in detailed source evidence;
they are no longer repeated in the README table or GitHub description. Private
scores do not inherit public percentiles. This supersedes the canary-2 presentation
format reviewed below; its historical regression and source-preservation review
remain dated evidence. Local reproducibility may use scientific functions, a
notebook, script or existing thin command entry point; no bespoke CLI is required.
The existing canary-2 releases and completed merge/branch cleanup remain intact.
The original profile/standard are retained at `v0.2.0-canary.2`; the current
profile receipt records the latest approved compact description.


Review date: **2026-11-09** · Acceptance standard: [Portfolio standard 1.1](portfolio-standard.md).

This audit covers the canary-2 `fix/portfolio-evidence` correction, the recorded
canary and the retained modeling/evaluation evidence. **Evidenced** means the
linked implementation or scoped check exists; **release gate** means the current
correction still needs final CI/merge/release confirmation. **Open P1** remains
accepted work. This document does not declare production readiness.

## The regression and its correction

The canary editorial change preserved the score data but weakened its visibility:

| Surface | Before the editorial change | Prior canary presentation | Canary-2 correction |
|---|---|---|---|
| README ranking paragraph | Line **5** at `b497e29` | Line **35** at `e55547f`, after model detail | A primary **Results** section immediately after the brief opening and before model/architecture/reproduction detail |
| GitHub description | Score/ranking context was a portfolio requirement | Replaced by a problem-only description; score, reference position/team count and Top % were absent | Live description again pairs historical and modern public scores with position/N and Top % |

The line movement is verified from Git blobs. GitHub descriptions are external
metadata; Git alone does not reconstruct their edit history. The restored live
value is captured in [public-profile.json](public-profile.json) and was checked
against GitHub during this review. Its historical entry is **0.43155,
82/1,572, Top 5.22%**; the modern DeBERTa entry is **0.35189, 1,193/1,572,
Top 75.89%**. The README additionally retains the historical final submission
**0.42654, 104/1,572, Top 6.62%**. The reference snapshot is
**2026-11-09T00:02:27Z**. [Reference receipt](leaderboard-context.json).

These are public **score-equivalent reference positions**, including insertion
positions and the receipt's tie treatment, not awarded final ranks. OOF/holdout
rank and Top % are not applicable; private rank is unavailable without a matching
private reference population. Restoring visibility does not transfer the stronger
historical score to the modern checkpoint.

## Requirement-by-requirement record

| ID | Priority | Status / applicability | Evidence and remaining acceptance |
|---|---|---|---|
| R01 | P0 | Evidenced; portfolio-wide review remains a release responsibility | All 26 requirements are mapped here against [standard 1.0](portfolio-standard.md). Branding or serving work cannot silently replace the modeling, ranking or production requirements. |
| R02 | P0 | Evidenced — solution | [README](../README.md) presents the Q&A problem, shared transformer, controlled input comparison, grouped evaluation, executable scripts and saved-model interface. |
| R03 | P0 | Verified compact presentation | Primary README and description retain public/private scores and public Top %. Rank/N and ties stay in linked detailed evidence. |
| R04 | P0 | Verified source scope | The metric/direction and submission are identified; private scores are paired with the same public submission. Detailed evidence retains reference counts, ties and snapshot. |
| R05 | P0 | Evidenced — comparison semantics | The primary Results display and detailed results explicitly distinguish score-equivalent from awarded rank, historical account metadata from a source-reproduced checkpoint, public/private populations and internal OOF/holdout. No CV percentile is inferred. |
| R06 | P0 | Evidenced — controlled ML insight | [Modeling study](modeling-study.md) compares title, body and answer inputs on three fixed grouped plans: aggregate mean OOF Spearman 0.220134 → 0.294286 → 0.325638, with an answer-family gain and a small question-family decline. It retains variation, a use-dependent decision and the next hypothesis. This is a Ridge experiment, not a claimed DeBERTa component gain. |
| R07 | P0 | Evidenced within the retained holdout; future causes unresolved | [Error analysis](error-analysis.md) traces question/answer inputs through the shared representation to supported answer-quality disagreements. It reports category/length support and a rare-label tie diagnostic, then proposes a controlled answer-side context experiment. It does not identify an unmeasured causal mechanism. |
| R08 | P0 | Evidenced with explicit scope limits | [CV receipt](cv-validation.json), [architecture](architecture.md) and [results](results.md) retain question-group boundaries, train-only Ridge vocabularies, exact assignments/IDs, target support, all-row Ridge OOF and independent metrics. The transformer evidence is one saved selected-checkpoint holdout, not five-fold transformer OOF. |
| R09 | P0 | Evidenced — diagnostic guardrails | [Hard-case analyzer](../scripts/analyze_hard_cases.py) reconciles all 1,216 rows and all 30 targets. Its fixed top-10% rank-gap cohort is retrospective; rank gap is not an additive Spearman contribution. No row/target removal, threshold tuning or new filtered headline is performed. |
| R10 | P0 | Evidenced — real saved model | [Canary receipt](canary-validation.json), [serving guide](serving.md) and [adapter](../src/quest/serving.py) bind the actual checkpoint, tokenizer, text lengths, target order, dependencies and package digest. Original evaluation/checkpoint provenance remains separate. |
| R11 | P0 | Evidenced for the recorded real-model canary | The canary checks native output → MLflow package → fresh-workspace HTTP inference: **2 real-input rows / 60 values**, maximum absolute error **0**, required-field rejection **HTTP 400**, package integrity and cleanup. It is not a load, authentication or full-validation benchmark. |
| R12 | P0 | Evidenced — declared text boundary | [Serving](serving.md) accepts `question_title`, `question_body` and `answer` strings and returns 30 named annotation scores. Tokenization/truncation is part of the package. The scores are not factual-truth probabilities; answer-dependent inputs are unavailable to an unanswered-question workflow. |
| R13 | P0 | Evidenced — maintained execution paths | [Reproduction](reproduction.md), Docker/dev-container files, CLI, executed notebooks and offline Kaggle entry points remain present. [Validation](validation.json) and [results](results.md) distinguish actual hosted training/scoring, CPU reload and container evidence. |
| R14 | P0 | Evidenced — client-owned integration | [Consumer contract](consumer-contract.md) assigns names, packages and scientific acceptance to this repo and links to generic [ML infrastructure](https://github.com/cttsai1985/ml-infrastructure). This solution does not require a platform-owned client catalog. |
| R15 | P0 | Delegated to shared infrastructure; client boundary reviewed | The shared `./infra` interface owns protected deployment state and lifecycle behavior. QUEST owns its model CLI/export and supplies its runtime explicitly. This audit does not claim to rerun infrastructure state/volume tests or require a duplicate deployment manager in the solution. |
| R16 | P0 | Evidenced for the scoped correction/canary | Existing `kaggle/quest` and artifact identities remain in the client contract. The recorded canary reports `shared_services_modified: false`; the correction changes presentation/analysis/metadata, not a live service deployment. No new live-service inventory is claimed by this document. |
| R17 | P0 | Evidenced for published diagnostic/canary scope | [Error receipt](error-analysis.json) contains aggregate counts, scores and hashes, not IDs/text/row predictions. The canary reports private assets unpublished; model packages, requests and logs remain outside Git. Release-asset inspection remains a publication gate. |
| R18 | P0 | Scoped evidence retained; current-source release gate | [Canary receipt](canary-validation.json) records clean adapter source, controller source, lock/runtime/package hashes for its actual earlier check. New error-analysis hashes match the analyzer and retained files. The correction version is **0.2.0.dev2**; its final CI/source/checksum record must not relabel the older HTTP run as freshly rerun. |
| R19 | P0 | Recorded prerelease exists; correction publication is a release gate | [v0.2.0-canary.1](https://github.com/cttsai1985/kaggle-google-quest-solution/releases/tag/v0.2.0-canary.1) is a prerelease targeting `e55547f`. A correction canary must identify its own source/lock and diagnostics. Canary means tested prerelease scope, not production traffic splitting. Other repositories have their own audits. |
| R20 | P0 | Evidenced — naming, attribution and solution-first opening | Canonical name is `kaggle-google-quest-solution`; the description combines the solution and transparent historical/current comparisons. The README leads with problem and measured input decisions. [Legacy source](../legacy/README.md) and [Apache-2.0 license](../LICENSE) remain. |
| R21 | P0 | Evidenced — primary-source domain/value context | [Problem and value](problem-and-value.md) cites the official Google competition/data description, explains subjective Q&A quality and separates proposed review/routing benefits from measured model results and unmeasured customer outcomes. |
| R22 | P1 | **Open P1 — production target retained** | The real-model endpoint canary proves a bounded interface/parity function. Production research/batch and online operation remain accepted goals in [shared requirements](https://github.com/cttsai1985/ml-infrastructure/blob/main/docs/requirements.md); this repo is not redefined as demo-only to close them. |
| R23 | P1 | **Open P1 — shared and model-specific operations** | Hosting/ownership, SLOs, model resources, TLS/authentication, request/service monitoring and alert delivery, off-host backup/restore and production rollout/rollback require actual deployment evidence. Current canary flags explicitly exclude production ingress, serving-image and zero-downtime proof. |
| R24 | P1 | Client recovery evidenced; production batch operation **open P1** | [Consumer contract](consumer-contract.md), [tracking tests](../tests/test_tracking.py) and import recovery preserve logical identity and verify artifacts before completion. The shared real saved-model fault drill used Iceberg; no QUEST scheduled production batch is inferred from it. |
| R25 | P1 | Current mining implemented; hypothesis validation **open P1** | [Error analysis](error-analysis.md) exactly reconciles retained mean Spearman, profiles all target families/domains and interprets rare-label ties. Its question-context/truncation and target-support follow-ups have not been fitted or prospectively validated. |
| R26 | P0 | Canonical URLs reviewed; final branch/release hygiene gate | [Public profile](public-profile.json) records the live `*-solution` URL. Canary-2 metadata/export URLs are reviewed separately from immutable old receipts. Retain the active `fix/portfolio-evidence` branch until merged and verified; delete only verified merged branches, preserving active worktrees/unmerged work. |

## Scientific preservation: exact comparison scope

Two distinct comparisons were inspected:

1. **Before the editorial/serving canary:**
   `b497e2953f5dc23159bdb86616c5f0e4ec1ff95f`.
2. **Published prior-canary merge:**
   `e55547ff5c63d21640157a89f74b2d1da1666183`.
3. **Canary-2 correction:** the working tree based on that prior-canary merge,
   reviewed before the final correction commit/release.

`git diff --name-status` is empty for the following path scope in both
before→canary and canary→current comparisons:

- `configs/`, `notebooks/`, `legacy/` and `LICENSE`;
- `src/quest/data.py`, `model.py`, `runtime.py`, `cv.py`,
  `tracking.py` and `import_support.py`;
- `docs/modeling-study.md`, `docs/modeling-study.json`, `docs/validation.json`,
  `docs/cv-validation.json` and `docs/leaderboard-context.json`.

The canary added `src/quest/serving.py` and changed package identity metadata in
`src/quest/__init__.py`; those additions are not described as preserved training
code. The canary-2 correction leaves the serving adapter unchanged, updates package
version metadata to `0.2.0.dev2`, and fixes the CLI exporter to read that declared
version instead of a hardcoded `0.1.0` wheel path. The CLI is therefore excluded
from the current unchanged-file scope. The local `.local/canary2-export-check`
check produced `quest_labeling-0.2.0.dev2-py3-none-any.whl`; it did not publish a
Kaggle asset, execute a hosted notebook or submit a score. It adds
navigation links to `docs/results.md` without altering its retained values.
`docs/canary-validation.json` is byte-unchanged from `e55547f`.

New error-analysis files are separate derived evidence. Their script/input/
prediction/split hashes, full-population metric reconciliation and stated scope
are inspectable; they do not rewrite the old modeling study or canary. Metadata,
URLs, README, CI/guard and package/lock version changes are intentional and reviewed
outside the protected scientific scope. This is not a blanket claim that the
entire repository is unchanged or that every historical revision has been audited.

## Explicit open work and release gates

- **Production operation — R22/R23/R24:** retain the shared platform's production
  target and complete its owner/SLO, access, capacity, alerting, backup/restore and
  deployment evidence. Agree this model's resource and review requirements in the
  intended workflow. A successful temporary HTTP request does not close these items.
- **Scientific follow-up — R25:** test answer-side question context under a fixed
  token/training budget and predeclared grouped evaluation; audit actual tokenizer
  truncation and obtain adequate rare-label support. Current associations are
  exploratory, not causal findings or proven improvements.
- **Correction publication — R18/R19/R26:** pass the complete current-source CI and
  portfolio guard, verify live description and source-backed Results, publish the
  appropriate canary/source/lock/checksums, then clean only verified merged branches.
  The old canary's source and metric evidence must retain their original identity.

## Metric-aware method restoration — canary 3

The original random-search rounding method and selective target application are restored in `src/quest/postprocessing.py`; archived source is unchanged. The protocol and code were committed at `fa949d2729d9707164bbba2f9473f07f6a8fa3e0` before fitting. `docs/postprocessing-study.json` retains source/input/split/policy hashes, all targets and folds, the nested Ridge assessment and explicitly selected-checkpoint DeBERTa diagnostics. Both fold mean and pooled OOF remain visible.

The saved-policy application helper validates target order and IDs, reloads JSON state, and writes separate predictions. It does not change raw model inference, hosted notebook scoring, the serving adapter, historical leaderboard values or prior receipts. Package metadata and lock now agree on `0.2.0.dev3`. The existing canary HTTP receipt retains its original source; it is not presented as a fresh serving test for this release. No fitted policy is promoted, and the independent transformer model-plus-policy acceptance gate remains open.

## Fixed DeBERTa transfer follow-up

The follow-up protocol and script were committed at
`2f9499e2d86b6116afa7df0f1e0809e2270aecbe` before fitting; the aggregate receipt's
full `source.git_commit` is authoritative. Five predeclared question-group seeds
produce ten bidirectional half-split calibration/assessment transfers. All ten
scores and all 30 targets remain in `docs/deberta-rounding-transfer.json`, including
the two negative transfers. Partition-rank normalization is retained as an
unsuccessful diagnostic, not silently promoted. The full-holdout canary-3 policy
was fixed before these results. Local snapshot application reproduces 476×30
candidate values exactly. No fitted policy, row-level data or candidate notebook
is published to Git; no new hosted execution, serving promotion or leaderboard
score is claimed. Existing historical/current leaderboard rows and old receipts
remain unchanged. The current walkthrough executes the aggregate follow-up.
