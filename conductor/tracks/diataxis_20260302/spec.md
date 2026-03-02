# Specification: Adopt the Diátaxis Framework for Documentation

## 1. Overview
Restructure the Locus documentation into the Diátaxis framework to ensure scalability and clarity as the team and user base grow. Information will be divided into four distinct physical quadrants: Tutorials (learning-oriented), How-To Guides (problem-oriented), Explanation (understanding-oriented), and Reference (information-oriented).

## 2. Functional Requirements
### 2.1 Physical Directory Restructuring
- Create new subdirectories in `docs/`: `tutorials/`, `how-to/`, `explanation/`, and `reference/`.
- Move existing files to their respective quadrants:
  - **Tutorials:** Move `guide.md` (e.g., "How to build your first perception pipeline with locus-tag") to `docs/tutorials/`.
  - **Explanation:** Move `architecture.md` and `coordinates.md` to `docs/explanation/`.
  - **Reference:** Move `api.md` to `docs/reference/`.
- Establish `docs/how-to/` with initial placeholders or content (e.g., "How to add a custom vendor fiducial dictionary").

### 2.2 Navigation & Index
- Update `mkdocs.yml` to reflect the new quadrant-based navigation structure.
- Ensure `docs/index.md` remains intact as a pointer/gateway to `README.md`.
- Retain `docs/engineering/` as internal documentation, integrating its navigation logically without cluttering the user-facing Diátaxis structure.

### 2.3 Link Validation
- Update internal relative links across all moved markdown files to ensure no broken references (e.g., fixing image links or cross-page links).

## 3. Acceptance Criteria
- [ ] `docs/` contains the four Diátaxis subdirectories.
- [ ] `guide.md`, `architecture.md`, `coordinates.md`, and `api.md` are moved to their correct locations.
- [ ] `mkdocs.yml` is successfully updated and the site builds without navigation errors (`uv run mkdocs build --strict`).
- [ ] All internal markdown links are correctly resolving.

## 4. Out of Scope
- Rewriting the actual content of the existing pages (beyond updating links or adding top-level headings).
- Modifying the underlying Sphinx/MkDocs theme configuration significantly.