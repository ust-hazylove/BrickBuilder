# Anonymous Release Checklist

Use this checklist before creating the public anonymous review repository.

## Required source files

- [ ] `README.md`
- [ ] `docs/REVIEWER_GUIDE.md`
- [ ] `requirements.txt`
- [ ] `environments.yaml`
- [ ] `app.py`
- [ ] `core_pipeline.py`
- [ ] `modules/`
- [ ] `ppo_repair/` source files
- [ ] `scripts/` source files needed for dataset, training, validation, and baselines
- [ ] `assembly_sequence/` source files if assembly-sequence experiments are included
- [ ] `Experiments/*.py` if paper metric scripts are included

## Required model/data artifacts for reviewer demo

- [ ] `weights/high_risk_predictor_styled_best.pt`
- [ ] `weights/ppo_lego_repair_final.zip`
- [ ] `image_inputs/` sample images
- [ ] `all_case_metrics.csv`
- [ ] `docs/text_comparison_prompts_v1.csv` for text-prompt comparison protocol

## Exclude from anonymous repo by default

- [ ] `output/`
- [ ] `logs/`
- [ ] `uploads/`
- [ ] `benchmark_output/`
- [ ] `comparison_output/`
- [ ] `qualitative_pack/`
- [ ] `qualitative_pack_100cases_20260417/`
- [ ] `paper/` and `paper_pics/`
- [ ] `img2build_linux.tar`
- [ ] `__pycache__/`, `.pytest_cache/`, local IDE folders
- [ ] Nested `.git/` directories inside third-party folders

## Anonymity checks

- [ ] No author names, affiliations, personal emails, ORCID IDs, or lab URLs in README/docs.
- [ ] No conference submission ID in app metadata, README, or docs.
- [ ] No local absolute paths or personal home-directory paths in reviewer-facing files.
- [ ] Git remote URL and repository owner are anonymous.
- [ ] Commit author name/email are anonymous for the initial upload.
- [ ] Paper PDFs or supplementary files are not duplicated in the code repo unless they are anonymized.

## Suggested commands

```bash
python scripts/prepare_anonymous_release.py --output anonymous_release --dry-run
python scripts/prepare_anonymous_release.py --output anonymous_release
cd anonymous_release
git init
git status --short
git status --ignored --short
```

If you initialize Git directly in the working checkout, run the same `git status` commands before `git add` and confirm that only source, docs, small sample assets, and intended checkpoints appear.
