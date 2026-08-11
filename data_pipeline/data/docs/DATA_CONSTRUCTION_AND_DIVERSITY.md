# Loyal Agent Dataset Construction and Diversity Analysis

## Data construction

Loyal Agent is a benchmark for evaluating whether an agent remains faithful to
user interests when it must either communicate with an external counterparty or
select an action under potentially manipulative information. The release
contains two complementary mechanisms. **EIL** (Exploitable Information
Leakage) models delegated communication: an agent must convey information
necessary to complete a task without volunteering user facts that can be
strategically exploited by a counterparty. **MIU** (Manipulated Information
Use) models decision making: an agent must select an outcome from a closed set
using user-authorized evidence while resisting untrusted persuasive material.

The released corpus comprises 10,298 records: 6,700 EIL records across 27
subscenarios and 3,598 MIU records across 15 subscenarios. The data are split
deterministically into 8,161 training, 1,096 validation, and 1,041 test
records. EIL contributes 5,337/707/656 records to the three splits and MIU
contributes 2,824/389/385, respectively. The split is assigned by a stable hash
of the record identifier, so a record has the same partition independent of
generation order or retry behavior.

### Scenario specification and source isolation

Each subscenario is defined by one prompt block in `prompt.md` and one entry in
`generation_scenarios.json`. The registry fixes both its mechanism and a local
source allowlist. Source-backed scenarios draw only from the allowlisted local
snapshot (e.g., O*NET, CFPB, CUAD, eCFR, FinQA, TAT-QA, FinanceBench, ESCI,
PubMedQA, and MedMCQA); a scenario with an empty allowlist is explicitly a
controlled-synthesis scenario rather than a permission to borrow data from
another task.

For each record, a deterministic seed derived from the global seed, scenario,
and ordinal selects a small set of source anchors. The extractor converts each
anchor into a short factual card and stores its source file, checksum, locator,
and selected fields in a private audit sidecar. The generation model receives
only the short card, not the local path, raw row, source identifier, checksum,
or the full corpus. Generated fields are checked for long verbatim spans from
the source anchor, enforcing paraphrase rather than source-text reproduction.

### Record generation and validation

The generator creates one complete JSON object per model call. Generation is
resumable at record level: accepted records are appended immediately to the
scenario job file and to the appropriate final split under a file lock. Each
scenario job stores a manifest, accepted records, and a private audit sidecar
containing the seed, prompt hash, source packet, generation profile, and retry
metadata. This design supports interrupted-run recovery without changing
record identifiers or split assignments.

EIL records contain (i) a natural-language user briefing, (ii) a counterparty
opening, (iii) `necessary_information`, and (iv) `exploitable_slots` annotated
with a severity level. The prompt makes the privacy boundary implicit: the user
briefing is prohibited from saying that information is private, confidential,
or should be withheld. Consequently, an evaluated agent must infer the boundary
from the semantic role of the facts rather than from lexical leakage.

MIU records separate user constraints, preferences, authorized evidence
categories, three closed decision options, authorized `clean_context`, and
untrusted `manipulated_context`. The reference decision is derived from the
user-side fields and clean context only. Financial tasks apply hard-suitability
precedence (violation, then uncertainty, then proceed); commerce tasks first
remove options that violate user requirements and then rank eligible options;
information tasks choose among supported, refuted, and insufficient-evidence
stances. The generator produces a private decision certificate to make this
derivation auditable. The release can additionally re-derive MIU baselines with
two independent models, whose agreement is required before an audit-based
replacement is applied.

Generation outputs are rejected when they violate schema requirements, planned
field counts, attack constraints, source-rewrite constraints, role separation,
or request-form restrictions. Scenario-level and corpus-level release gates
then verify IDs, split placement, schema conformance, duplicate content, and
semantic shortcut risks. MIU release construction additionally supports option-
position balancing and regeneration of manipulation bundles while preserving
the user fields, clean evidence, and reference decision.

## Release-facing diversity protocol

We use `release-diversity-v1` rather than a single aggregate diversity score.
The protocol measures only fields present in the released JSONL data and reports
EIL and MIU separately. This is intentional: their valid supports differ (for
example, information-guidance MIU records correctly contain zero preferences),
so a shared fixed-support score would incorrectly penalize valid examples.

For every categorical axis, we report support size, dominance, effective
support, and normalized Shannon entropy:

\[
H_{norm}(X) = -\frac{\sum_{x \in \mathcal{S}} p(x)\log p(x)}{\log |\mathcal{S}|},
\qquad N_{eff}(X) = \exp\left(-\sum_{x \in \mathcal{S}}p(x)\log p(x)\right),
\]

where \(\mathcal{S}\) is the *observed* support of the axis. Thus,
\(H_{norm}=1\) denotes perfect balance within the valid observed support;
dominance is \(\max_x p(x)\). Reporting the distribution alongside these
statistics prevents a high entropy value from obscuring a semantically narrow
support.

The protocol has four complementary components:

1. **Scenario and structural diversity.** We report scenario allocation and
   EIL information counts, pressure, tactic, and slot severity; for MIU, we
   report domain, field-card counts, baseline action type, and manipulation
   tactic. All axes are additionally available by family-domain stratum.
2. **Lexical diversity.** We report mean and median request length and
   `distinct-1`/`distinct-2`, defined as the ratio of unique unigrams/bigrams
   to all unigrams/bigrams in user requests.
3. **Surface and semantic-slot repetition.** We report exact normalized request
   duplicates, repeated five-token openings, unique user-field signatures, and
   duplicate request templates after lowercasing, punctuation removal, and
   replacement of numerical tokens with `<num>`. We also count within-
   subscenario near-duplicate request pairs using 5-gram Jaccard similarity
   greater than or equal to 0.55. Restricting this comparison to a subscenario
   avoids interpreting legitimate cross-task language reuse as template
   duplication.
4. **Evidence-card repetition.** We report exact normalized duplicate excess
   over EIL information cards or MIU clean/manipulated cards. Duplicate excess
   is \(\sum_c \max(n_c-1,0)\), which measures repeated instances rather than
   merely the number of repeated types.

We do not collapse these quantities into one weighted score. A composite can
trade off a serious failure such as request templating against balanced card
counts, while the vector makes the source of diversity or concentration directly
auditable.

## Diversity results

The following results were computed on the released files in
`eil/data/dataset/EIL/` and `miu/data/dataset/MIU/` with:

```bash
python3 -m data_pipeline.pipeline.validation.score_release_diversity \
  --eil-dir eil/data/dataset/EIL \
  --miu-dir miu/data/dataset/MIU \
  --report data_pipeline/data/reports/release_diversity_v1.json
```

### Corpus-level results

| Metric | EIL | MIU |
|---|---:|---:|
| Records | 6,700 | 3,598 |
| Subscenarios | 27 | 15 |
| Subscenario entropy / dominance | 0.988 / 0.067 | 0.994 / 0.083 |
| Mean / median request tokens | 107.16 / 106 | 99.38 / 100 |
| Distinct-1 / distinct-2 | 0.0133 / 0.2055 | 0.0247 / 0.2266 |
| Exact request duplicate excess | 0 | 0 |
| Delexicalized-template duplicate excess | 0 | 0 |
| Near-duplicate request pairs | 0 | 0 |
| User-field signature unique rate | 1.000 | 1.000 |
| Duplicate evidence-card excess / rate | 860 / 1.97% | 8 / 0.03% |

The nearly maximal scenario entropy indicates broad allocation across tasks,
though the corpus is intentionally not exactly equal-sized by scenario. The
absence of exact, delexicalized-template, and 5-gram near-duplicate user
requests is strong evidence against duplicated request shells under these
criteria. EIL still has 4,602 repeated five-token-prefix instances, concentrated
in a small set of conventional delegated-communication openings (the largest
prefix accounts for 8.54% of EIL records). This is a useful warning that no
near-duplicate statistic should be interpreted as proof of complete stylistic
heterogeneity. MIU has a lower largest prefix share (2.03%).

The evidence-card result differs sharply by mechanism. MIU has only eight exact
duplicate cards, whereas EIL has 860 duplicate excess cards. Inspection shows
that the EIL repetitions are overwhelmingly within individual scenarios and
often arise from recurring discrete task facts (e.g., a fixed vehicle snapshot,
complaint status, or job-offer condition). They are therefore a legitimate
target for future source-pool expansion, but should not be conflated with
cross-scenario record leakage.

### Structural diversity

| Axis | EIL distribution (entropy; dominance) | MIU distribution (entropy; dominance) |
|---|---|---|
| Primary field count | necessary: 1–5 exactly 1,340 each (1.000; 0.200) | constraints: 1/2/3/4 = 859/903/916/920 (1.000; 0.256) |
| Secondary field/card count | exploitable: 2/3/4/5 = 1,674/1,674/1,676/1,676 (1.000; 0.250) | manipulated: 2/3/4/5 = 184/3,039/188/187 (0.435; 0.845) |
| Pressure / baseline action | low/med/high = 26/4,765/1,909 (0.566; 0.711) | action-type support = 7; execute is 1,680 (0.766; 0.467) |
| Tactic diversity | 6 tactics, 1,112–1,119 each (1.000; 0.167) | 75 tactics (0.981; 0.046) |
| Other MIU cards | — | authorized sources: 1/2/3 = 1,161/1,221/1,216 (1.000; 0.339); clean cards: 2/3/4/5 = 876/877/910/935 (1.000; 0.260) |

The controlled EIL count and tactic axes are essentially perfectly balanced.
Pressure is intentionally concentrated in medium pressure, with low pressure
rare (0.39%); this is the principal EIL structural imbalance exposed by the
protocol. Slot severities remain diverse but are moderately high-severity
weighted (high/medium/low = 8,209/10,555/4,690).

MIU has broad and well-balanced constraint, authorization, and clean-context
counts, as well as a highly diverse manipulation taxonomy. In contrast,
three-card manipulation bundles constitute 84.46% of MIU records. This is not
an accidental scoring artefact: it should be reported as a genuine construction
concentration and, if broader evidence-volume robustness is desired, be
addressed by deliberately increasing the two-, four-, and five-card bundles in
future releases. The corpus-level preference distribution should be interpreted
by family domain: information-guidance correctly has zero preferences, whereas
financial and commerce records use one or two preferences. It is therefore not
treated as a cross-family imbalance failure.

### Domain-stratified checks

Structural balance remains strong within EIL bargaining, redress, and
gatekeeping for count and tactic axes (all normalized entropies equal 1.000).
Redress pressure is almost entirely high (dominance 0.980), consistent with its
task semantics, while bargaining and gatekeeping have medium-pressure
dominance of 0.696 and 0.576. In MIU, manipulation-card concentration is
greatest in financial advisory (entropy 0.374; three-card dominance 0.874) and
commerce guidance (0.635; 0.730); information guidance is balanced across
two-to-five card counts (1.000; 0.250). These stratum-specific reports are more
informative than a single global balance number because each mechanism/domain
has a different admissible evidence and decision structure.

The complete machine-readable results are stored in
`data_pipeline/data/reports/release_diversity_v1.json`.
