#!/bin/bash
# Tag current HEAD (last feature commit), update _version.py, push tag + GitHub release.
set -e

DATE=$(date +%Y%m%d)
HASH=$(git rev-parse --short HEAD)
GIT_TAG="${DATE}-${HASH}"       # git tag: dash (+ invalid in tags)
VERSION="${DATE}+${HASH}"       # PEP 440: + for local segment

echo "Version: $VERSION  (tag: $GIT_TAG)"

# Tag current HEAD (last feature commit) BEFORE version bump commit
if git rev-parse "$GIT_TAG" >/dev/null 2>&1; then
    echo "Tag $GIT_TAG already exists — deleting and re-tagging"
    git tag -d "$GIT_TAG"
fi
git tag "$GIT_TAG"
echo "Tagged HEAD as: $GIT_TAG"

# Update _version.py
VERSION_FILE="$(dirname "$0")/../darkbottomline/_version.py"
cat > "$VERSION_FILE" <<EOF
"""Package version information for DarkBottomLine."""

__version__ = "${VERSION}"
EOF

echo "Updated _version.py"

# Commit version bump (after tag — tag stays on feature commit)
git add "$VERSION_FILE"

BODY_MSG_FILE=$(mktemp)
trap 'rm -f "$BODY_MSG_FILE"' EXIT
cat > "$BODY_MSG_FILE" <<EOF
# Optional commit body — appended below "chore(version): bump to ${GIT_TAG}".
# Lines starting with '#' are ignored. Save + quit empty to skip.
EOF
"${EDITOR:-vi}" "$BODY_MSG_FILE"
BODY_MSG=$(grep -v '^#' "$BODY_MSG_FILE" | awk 'NF{p=1} p')

if [[ -n "$BODY_MSG" ]]; then
    git commit -m "chore(version): bump to ${GIT_TAG}" -m "$BODY_MSG"
else
    git commit -m "chore(version): bump to ${GIT_TAG}"
fi

# Push commit + tag + GitHub release
read -rp "Push to origin and create GitHub release? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    git push origin HEAD
    git push origin "$GIT_TAG"
    RELEASE_NOTES="Release ${GIT_TAG}"
    if [[ -n "$BODY_MSG" ]]; then
        RELEASE_NOTES="${RELEASE_NOTES}

${BODY_MSG}"
    fi
    gh release create "$GIT_TAG" \
        --title "$GIT_TAG" \
        --notes "$RELEASE_NOTES" \
        --target "$(git rev-parse ${GIT_TAG}^{})"
    echo "GitHub release created: $GIT_TAG"
else
    echo "Skipped. Run manually:"
    echo "  git push origin HEAD && git push origin $GIT_TAG"
    echo "  gh release create $GIT_TAG --title $GIT_TAG --notes 'Release $GIT_TAG'"
fi
