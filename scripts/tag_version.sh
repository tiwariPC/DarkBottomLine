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
"${EDITOR:-vi}" "$BODY_MSG_FILE"
BODY_MSG=$(cat "$BODY_MSG_FILE")

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

    # Previous tag (most recent tag before this one) — commit log between it
    # and this tag's target commit becomes the auto-generated changelog.
    PREV_TAG=$(git tag --sort=-creatordate | grep -v "^${GIT_TAG}$" | head -1)
    TARGET_SHA="$(git rev-parse ${GIT_TAG}^{})"
    if [[ -n "$PREV_TAG" ]]; then
        CHANGELOG=$(git log "${PREV_TAG}..${TARGET_SHA}" --pretty=format:'- %s' --no-merges)
    else
        CHANGELOG=$(git log "${TARGET_SHA}" --pretty=format:'- %s' --no-merges)
    fi

    RELEASE_NOTES="Release ${GIT_TAG}"
    if [[ -n "$BODY_MSG" ]]; then
        RELEASE_NOTES="${RELEASE_NOTES}

${BODY_MSG}"
    fi
    if [[ -n "$CHANGELOG" ]]; then
        RELEASE_NOTES="${RELEASE_NOTES}

## Changes since ${PREV_TAG:-start}
${CHANGELOG}"
    fi
    gh release create "$GIT_TAG" \
        --title "$GIT_TAG" \
        --notes "$RELEASE_NOTES" \
        --target "$TARGET_SHA"
    echo "GitHub release created: $GIT_TAG"
else
    echo "Skipped. Run manually:"
    echo "  git push origin HEAD && git push origin $GIT_TAG"
    echo "  gh release create $GIT_TAG --title $GIT_TAG --notes 'Release $GIT_TAG'"
fi
