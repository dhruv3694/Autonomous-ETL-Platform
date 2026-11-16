# A Local-First, Human-Guided System for Natural-Language ETL Planning

**Author(s):** [Insert author name(s)]  
**Affiliation(s):** [Insert institution or organization]  
**Contact:** [Insert corresponding author email]  
**Manuscript type:** System design and implementation paper

## Abstract

Extract, transform, and load (ETL) pipelines remain costly to build and maintain because source datasets vary in format, schema, quality, and business meaning. Natural-language interfaces and large language models (LLMs) may reduce the effort required to specify transformations, but unrestricted generation and execution of code creates correctness, safety, and auditability risks. This paper presents a local-first ETL workflow application that translates natural-language requests into reviewable transformation plans. The system combines dataset profiling, heuristic schema intelligence, optional locally configured model assistance, deterministic fallback planning, an allowlist of supported operations, human approval gates, Python and SQLite execution paths, validation, and workflow history.

The planner uses bounded dataset context rather than including complete tables in model prompts. Model output is treated as an untrusted plan: it is validated against available datasets, columns, dependencies, and supported operations before previews or execution are allowed. Users inspect and approve operations before execution. The SQL workflow uses generated SQLite statements rather than executing model-authored SQL. An optimization studio reports query-plan evidence separately from timing measurements, checks output equivalence, and requires explicit approval before benchmarking candidates.

The implementation is evaluated through automated tests covering planning, schema analysis, approval persistence, SQL execution, output equivalence, and failure handling. The test results establish software-behavior checks, not comparative performance or model-accuracy claims. The project is therefore best understood as an implemented research prototype and demonstration system. Its primary contribution is an end-to-end design that places human review and deterministic validation between natural-language intent and data modification.

**Index Terms—** ETL, data integration, schema matching, natural-language interfaces, human-in-the-loop systems, local AI, data quality, SQLite.

## I. Introduction

Data integration projects routinely combine files that differ in schema, naming conventions, missing-value patterns, and representations of dates or categories. Creating a dependable ETL pipeline consequently requires both technical implementation and domain interpretation. Schema matching research has long treated the correspondence between fields as a central problem in integrating heterogeneous sources [1]. In practice, users also need to express transformation intent, inspect its interpretation, and determine whether the resulting data is fit for use.

Natural-language planning can make pipeline authoring more accessible, but it does not by itself provide trustworthy execution. A generated transformation may reference a nonexistent field, misunderstand a business term, or change values in an unintended way. Executing free-form generated code or SQL increases the consequences of such errors. These concerns motivate a design in which model assistance is constrained to planning and a transparent workflow retains control over validation and execution.

This paper describes a local-first ETL application that supports dataset upload, schema profiling, natural-language plan generation, review and approval, execution, validation, comparison, and performance inspection. Its intended role is decision support for ETL authoring rather than autonomous, unsupervised integration.

The system makes four practical contributions:

1. A workflow that turns user instructions and bounded dataset profiles into structured, reviewable transformation plans.
2. A constrained planning boundary: supported operations, real column names, dependency checks, and clarification handling are validated before execution.
3. Human approval and audit records integrated into execution rather than added as optional reporting.
4. Python and SQLite execution paths, accompanied by data validation and an SQL analysis workflow that distinguishes structural query plans from measured runtime.

The paper reports the implemented behavior and automated test coverage. It does not claim that the system outperforms other ETL tools, achieves a particular mapping accuracy, or improves runtime in general; those claims require controlled experiments that are outside the current evaluation.

## II. Background and Design Motivation

Schema matching seeks correspondences between attributes in different schemas. Approaches commonly combine evidence from names, types, instances, and domain knowledge [1]. This system follows that general pattern in a deliberately small, inspectable heuristic: exact and normalized name similarity, static semantic tags, logical types, and bounded sample overlap contribute evidence for candidate mappings.

The project also addresses a distinct engineering concern: natural-language intent is ambiguous, while a data transformation is executable and may be irreversible after export or downstream use. The application therefore separates plan generation from plan execution. It does not treat an LLM response as executable source code. Instead, it validates a structured representation, renders previews from supported operations, and requires user approval.

These are system-design choices, not claims that the application eliminates errors or guarantees privacy. The bounded context reduces the amount of data sent to a configured inference component, but sampled values can still be sensitive. Users remain responsible for choosing suitable datasets and configuring an appropriate local model environment.

## III. Requirements and Scope

The system was designed around the following requirements:

- **Dataset awareness:** infer useful metadata from selected datasets before planning.
- **Bounded context:** avoid placing complete dataset contents in the model prompt.
- **Constrained transformations:** accept only known operations supported by the executor.
- **Reviewability:** show the proposed workflow and operation previews before execution.
- **User authority:** require approval before applying transformations.
- **Failure transparency:** stop failed SQL execution and retain structured failure information rather than presenting a partial result as complete.
- **Traceability:** retain workflow events and execution details for review.
- **Local-first availability:** support deterministic planning when optional local model inference is unavailable.

The natural-language ETL executor currently supports duplicate removal, missing-value handling, date conversion, text cleanup, column-name normalization, and validation-only operations. Its workflow processes selected datasets independently. It does not claim to support arbitrary joins, aggregations, filters, derivations, free-form SQL, or all business-specific category mappings.

## IV. System Architecture

### A. Application Workflow

The application is implemented as a Streamlit user interface over Python ETL services. The main workflow is:

1. Upload one or more supported files and preview their contents.
2. Select datasets and enter a natural-language transformation request.
3. Profile each dataset and build bounded schema context.
4. Generate a structured plan using the selected model provider, or use a labeled deterministic fallback when model inference is unavailable or invalid.
5. Validate the plan and display operations, dependencies, risk classifications, and Python or SQL previews.
6. Allow operation-level review and approval, or approval of the complete pipeline.
7. Execute approved operations in Python or SQL mode.
8. Report audit details, validation results, before-and-after comparisons, and performance measurements.

The workflow state layer records plan generation, approval decisions, and execution events. Mapping-plan approvals are also persisted with reviewer metadata when supplied.

### B. Dataset Profiling and Context Construction

Before natural-language planning, the system builds a profile from the selected DataFrames. For each dataset, the context includes dataset identifiers and labels, format, row and column counts, duplicate-row and null counts, possible key columns, and per-column metadata. Column profiles can include dtype, null and unique counts, up to three short sample values, numeric summary statistics, candidate date information, and frequent values for low-cardinality columns. The planner limits the number of selected datasets and profiled columns and reports when columns are truncated.

The context builder also records possible inter-dataset relationships based on matching column names and overlap between bounded sample values. These are explicitly candidates rather than verified relationships. Full DataFrames are not serialized into the planner context.

The separate schema-intelligence module provides logical type inference and semantic tags. It includes a static synonym dictionary for common concepts such as customer identifiers, names, email addresses, dates, and amounts. Schema profiles and planner contexts are related but distinct outputs: the precise fields available depend on which component is invoked.

### C. Candidate Mapping

For source and target columns, the schema-mapping component scores each unused target as a potential match. Evidence includes exact names, normalized name similarity, shared semantic tags, equal logical types, and Jaccard overlap between bounded sample sets. In the implementation, normalized similarity contributes a weighted score; semantic-tag evidence can increase the score; matching logical types and sufficiently high sample overlap provide additional boosts. The resulting score is categorized as HIGH, MEDIUM, LOW, or INCOMPATIBLE, with lower-confidence matches requiring review. Identifier matches without semantic evidence are downgraded to avoid treating generic identifier resemblance as sufficient proof.

These scores are heuristic confidence indicators, not calibrated probabilities. They should not be interpreted as measured accuracy. The user must verify mappings against domain meaning, especially for fields with similar names but different business semantics.

### D. Natural-Language Planning

The planner accepts an instruction and bounded, data-derived context. When configured inference is available, a model is asked for a structured plan. The response is parsed and validated against the actual selected dataset identifiers, column names, supported operations, dependency structure, and ambiguity constraints. Unsupported or unclear requests can be reported as clarification items and block execution rather than being silently reinterpreted.

If model inference is unavailable or its output fails validation, the system can use a deterministic heuristic fallback. The application labels the fallback distinctly; it does not present it as successful model inference. This allows basic planning functionality without an external paid inference service. Hugging Face model support is configured for local-cache use as documented by the project; the required weights must be available locally.

The plan is an intermediate representation, not generated executable code. Python and SQL previews are rendered from known operations. The model cannot introduce arbitrary imports, arbitrary Python, or executable raw SQL through the structured planning path.

### E. Approval and Execution

Execution is separated from planning. Pipeline execution checks for approval, and the interface supports operation-level as well as pipeline-level approval. The Python engine applies supported operations to copies of input DataFrames, records per-step audit and performance information, and validates intermediate and final results.

The SQL path translates supported operations into SQLite statements. It uses generated internal column identifiers, source-aware SQLite affinities, bound inserts, and output-label restoration to address special source-column labels and type round trips. Model-produced raw SQL is rejected. Python and SQL semantics can differ for some operations, so the two modes are not claimed to be interchangeable.

### F. Validation, Comparison, and Failure Handling

The execution layer checks row changes, duplicate rows, nulls, and empty output schemas, and records issues in validation results. The application also provides before-and-after comparison and transformation impact views.

SQL failures are represented as structured errors with execution and step identifiers, operation type, error category, timestamp, redacted SQL context, and audit history. Failed execution stops the workflow from treating a partial output as a successful final result. SQL literal redaction is intended to limit accidental disclosure in error context; it is not a general-purpose secret-detection mechanism.

### G. SQL Optimization Studio

The SQL analysis component runs SQLite `EXPLAIN QUERY PLAN` and presents its output as structural evidence. It does not create indexes automatically and does not infer general performance gains from query-plan structure alone. The built-in candidate generator currently removes generated validation-only projection steps, which are intended as no-ops in that generated SQL workflow.

Candidates require explicit approval before repeated benchmarking. Equivalence checks compare column labels and order, dtypes, shape, and row multisets including nulls and duplicate multiplicities. Benchmark reports distinguish improvement, regression, inconclusive results, and rejected non-equivalent outputs. Measurements include profiler overhead and short-run timings may be inconclusive. The optional local advisor is advisory; it cannot change executable SQL.

## V. Implementation

The project is written in Python. Pandas provides the primary in-memory tabular representation; SQLite supports the SQL execution path and query-plan inspection; Streamlit provides the user interface. The application supports several common file formats, including delimited text, JSON variants, XML, Excel, Parquet, and SQLite inputs, as described in the project documentation.

The code is organized into ETL modules for data loading, schema intelligence, pipeline planning, execution, SQL generation and metadata, comparison, profiling, model configuration, and workflow persistence. The UI brings these services together for dataset selection, model configuration, plan review, execution, comparison, and performance inspection.

Performance reporting includes stage and operation durations, CPU time, sampled process RSS, Python allocation estimates, and throughput. These measurements have explicit limitations: process RSS includes the Streamlit process, Python allocation tracking does not capture all native Pandas or NumPy memory, system disk and network counters are not process-attributed, and GPU utilization is not reported as available.

## VI. Evaluation

### A. Evaluation Method

The implementation was evaluated with the repository’s automated test suite. Tests include schema profiling and mapping, natural-language plan structure and execution, approval enforcement, persistence of mapping plans and decisions, isolation checks, Python and SQL execution behavior, SQL failure handling, output equivalence, optimizer candidate validation, and performance-profiler behavior.

The natural-language pipeline tests use a mocked inference response for deterministic checks. They verify, among other behaviors, that the planner uses a bounded dataset profile, respects dependency order, rejects unapproved execution, applies approved allowlisted operations, and does not mutate the source DataFrame. SQL tests cover generated SQL validation, preservation of column labels and dtypes, duplicate and null handling, query-plan reporting, and candidate equivalence.

After fixes to schema inference and mapping-approval persistence, the project’s existing test suite completed successfully in the development environment. This result is evidence that the checked behaviors passed the current tests. It is not a substitute for an independent benchmark, a user study, formal verification, or broad compatibility testing.

### B. What the Evaluation Establishes

The tests provide evidence for the following software properties within their tested cases:

- Profile and mapping outputs have expected structures and representative classifications.
- A mocked structured model response can produce a validated plan for supported operations.
- Approval is required before execution in tested pipeline paths.
- Supported operations execute and produce expected outputs in selected scenarios.
- SQL generation rejects raw SQL and invalid dependencies in tested cases.
- SQL round-trip and equivalence logic detect relevant differences in labels, types, shape, nulls, and duplicate counts.
- Mapping approvals and execution metadata can be persisted and reloaded.
- The optimizer does not treat structural query-plan output as runtime evidence or create indexes automatically.

The tests do not establish end-user task completion rates, mapping precision or recall, model reliability across prompts, production-scale throughput, security against all adversarial inputs, or superiority to existing ETL systems.

### C. Reproducibility

The project can be installed from its dependency manifest and launched with Streamlit:

```text
pip install -r requirements.txt
streamlit run app.py
```

The automated checks can be run with:

```text
python -m pytest -q
```

Reproducing model-assisted results additionally requires the configured local model provider and model weights or service to be available. The project’s deterministic fallback and mocked tests do not require live model inference.

## VII. Discussion

The central design choice is to keep generative assistance upstream of a deterministic execution boundary. A model may help interpret an instruction, but it does not directly supply code or SQL to run. The allowlist and schema checks constrain the plan, while review gates leave the decision to execute with the user. This provides a concrete path to making a natural-language interface inspectable without representing model output as inherently correct.

The architecture also acknowledges that bounded context is a trade-off. Limiting the amount of data exposed to inference reduces unnecessary data transfer, but can reduce the evidence available for interpreting unusual values or business rules. A small sample is not a representative sample guarantee, and user-provided examples or domain metadata may be necessary for reliable interpretation.

Likewise, heuristic schema scores are useful for ranking candidates but cannot determine semantic equivalence by themselves. A column named “amount” could refer to a gross amount, tax amount, or balance. The system’s approval and clarification steps are therefore essential parts of the design, rather than optional interface conveniences.

## VIII. Limitations and Threats to Validity

**Limited empirical evaluation.** The current evaluation is based on automated behavioral tests. No benchmark dataset, baseline comparison, controlled performance experiment, or user study is reported. The paper therefore makes no quantitative claim about accuracy, productivity, or speed.

**Constrained operation set.** The supported transformations cover common cleaning tasks but do not constitute a general ETL language. Joins, complex filters, arbitrary derivations, aggregation, and business-specific category corrections are outside the documented executor scope.

**Heuristic mapping.** Static synonyms and similarity measures are domain-limited. Sample overlap may be misleading, and confidence categories are not statistically calibrated.

**Inference and data sensitivity.** Bounded prompts may include column names, statistics, and short sample values. Those values may contain personal, confidential, or regulated information. Local inference configuration does not by itself guarantee secure handling; deployment and data-governance controls remain necessary.

**Runtime and scale.** Pandas-based in-memory execution may be unsuitable for datasets that exceed available memory. The profiling and performance metrics have documented sampling and attribution limits. SQL execution uses SQLite and does not imply distributed or warehouse-scale processing.

**Cross-mode differences.** Python and SQLite can differ in type coercion, date parsing, and string behavior. The implementation documents this and does not claim semantic equivalence across execution modes in general.

**Security scope.** The allowlist and generated SQL design reduce specific classes of unsafe execution in the intended pipeline path. They do not constitute a complete security audit or prove the whole application secure.

## IX. Future Work

Future work should evaluate the system with a published, reproducible set of source-target schemas and natural-language transformation tasks. Relevant measurements would include schema-mapping precision and recall, plan validity, unsupported-request detection, human correction rate, task completion time, and execution equivalence. Baselines should include manual mappings and non-generative heuristic planning.

Additional engineering work could extend the operation set through versioned, typed transformation definitions; improve configurable domain vocabularies; support explicit user-provided mappings and examples; and broaden validation across file formats and larger datasets. Security and governance work should include a dedicated threat model, deployment-specific privacy controls, and independent review of data handling. Any expanded SQL optimization should retain semantic-equivalence checks and report repeated measurements with uncertainty rather than promising general speedups.

## X. Conclusion

This paper presented a local-first, human-guided ETL application that turns natural-language requests into constrained, reviewable transformation plans. The implementation combines bounded dataset profiling, heuristic schema matching, optional model-assisted planning, deterministic fallback, explicit approval, Python and SQLite execution, validation, audit history, and cautious SQL analysis.

The project demonstrates an implementable design for inserting validation and human control between AI-assisted planning and data modification. Its current tests support selected behavioral claims, while its limitations make clear that it is a prototype rather than a benchmark-validated general ETL automation system. Further evaluation on representative datasets and user tasks is necessary before making stronger claims about effectiveness or production readiness.

## References

[1] E. Rahm and P. A. Bernstein, “A survey of approaches to automatic schema matching,” *The VLDB Journal*, vol. 10, no. 4, pp. 334–350, Dec. 2001, doi: [10.1007/s007780100057](https://doi.org/10.1007/s007780100057).

## Acknowledgment

The authors should disclose the use of AI-assisted development or writing according to the target institution’s, venue’s, and funder’s policies. Replace this section with the required disclosure and acknowledgments before submission.
