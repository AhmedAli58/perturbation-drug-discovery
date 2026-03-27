# Problem Definition

## Industry Problem

Wet-lab perturbation screens are expensive, slow, and capacity-constrained. Teams working on target discovery or assay development need a reproducible way to:

- benchmark whether a model adds value over simple biological baselines
- estimate perturbation response patterns before committing new experiments
- prioritize which perturbations deserve follow-up screening

This repository addresses the engineering side of that workflow for a single public single-cell CRISPR dataset.

## Product Goal

Build a reproducible perturbation-response benchmark and ranking pipeline that:

- uses fixed dataset preparation rules
- separates known-perturbation and held-out-perturbation evaluation
- produces benchmark artifacts with clear provenance
- ranks candidate perturbations for wet-lab follow-up

## Intended Users

- ML engineers building assay-support or model-benchmarking systems
- computational biologists who need reproducible benchmark outputs
- hiring reviewers evaluating end-to-end ML engineering execution

## Non-Goals

This project does not attempt to:

- predict compound activity
- perform target-to-drug mapping
- estimate clinical efficacy or safety
- serve as a production decision engine without experimental validation

## Success Criteria

The project is successful when:

- data preparation is deterministic and versioned by manifest
- held-out perturbation evaluation is leakage-resistant
- all model claims are reported against naive and nearest-seen baselines
- candidate ranking outputs include provenance, status, and confidence
- a new reviewer can reproduce the benchmark workflow from documented commands
