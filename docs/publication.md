# Public files and local records

The public repository contains source, tests, reports, and sanitized evidence.
It does not contain game assets, model weights, raw recordings, or machine configuration.

## Public evidence

Keep the following facts when publishing experiment results:

- Record the software and package versions.
- Record the artifact filename, size, and digest.
- Record the source revision and public source URL.
- Record the test method, sample count, result, and limits.
- Distinguish measured results from untested functions.

Remove machine paths, account names, hostnames, process IDs, and window IDs from public reports.
Remove private server addresses and credentials.
Do not copy raw desktop titles into public evidence.
Use repository-relative paths in instructions.
Use asset filenames when the local installation path is irrelevant.

## Local files

Use `.local/` for installation details, external research source, and local configuration.
Use `runs/` for raw logs, screenshots, and traces.
Use `weights/` or `.local/weights/` for model files.
The Git ignore file excludes these directories.
The Git ignore file also excludes game archives and common model formats.

Raw local records can contain machine paths and window information.
The public report must not link to those excluded records.
Publish a sanitized result summary when the raw record is necessary evidence.
Keep the raw record locally for comparison.

## Check the public file set

Stage the intended files before the check.
Run this command before a public commit:

```bash
python3 scripts/check_public_files.py
```

The check reads the staged content when a Git index exists.
Before repository initialization, it examines the explicit project file set.
It checks common machine paths, credential patterns, and excluded artifact types.
It also rejects links from public documents to local run records.
The check cannot identify every possible secret or personal detail.
Review the staged changes before publication.

The current sanitization preserves the original test outcomes.
It does not upgrade an untested function to a tested function.
