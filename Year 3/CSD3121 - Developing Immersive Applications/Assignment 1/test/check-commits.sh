#!/usr/bin/env bash
set -euo pipefail

# This script checks whether the student has pushed at least 7 commits.
# Student ID is inferred from the repo name: ipa1-<student_id>.

repo_root=$(git rev-parse --show-toplevel)
repo_name=$(basename "$repo_root")
student_id=${repo_name#ipa1-}

if [[ -z "$student_id" || "$student_id" == "$repo_name" ]]; then
	echo "Unable to derive student ID from repo name '$repo_name'. Expected format: ipa1-<student_id>." >&2
	exit 1
fi

# Count commits authored by this student ID (case-insensitive match on author name/email).
commit_count=$(git -C "$repo_root" log --author "$student_id" --pretty=oneline | wc -l | tr -d ' ')

if [[ "$commit_count" -ge 7 ]]; then
	echo "PASS: Found $commit_count commits authored by '$student_id' (>= 7 required)."
	exit 0
fi

# Fallback: total commit count, in case the student ID is not present in author metadata.
total_count=$(git -C "$repo_root" rev-list --count HEAD)

if [[ "$commit_count" -lt 7 ]]; then
	echo "FAIL: Only $commit_count commits found with author matching '$student_id'. Need at least 7."
	echo "      Total commits in repo: $total_count (may include starter commits)." >&2
	echo "      Ensure your git author name/email includes your student ID or push at least 7 commits." >&2
	exit 1
fi
