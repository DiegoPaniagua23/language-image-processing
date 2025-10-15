# Advanced Repository Setup Guide

This document explains the advanced recommendations for improving your statistical computing repository with modern development practices.

## 📦 1. Package Management with `renv`

### What is `renv`?
`renv` is R's answer to Python's `virtualenv` or Node.js's `package.json`. It creates reproducible environments by:
- Isolating package libraries per project
- Recording exact package versions
- Allowing easy environment recreation

### Benefits:
- **Reproducibility**: Anyone can recreate your exact R environment
- **Collaboration**: Team members get identical package versions
- **Time travel**: Revert to previous package states
- **Isolation**: Different projects can use different package versions

### How to Use:

1. **Initial Setup** (run once per project):
```r
# In your project root
source("homework/01_poisson_logistic_regression/setup_renv.R")
```

2. **Day-to-day workflow**:
```r
# Install new packages
renv::install("new_package")

# Update lockfile after installing packages
renv::snapshot()

# Restore environment (for collaborators)
renv::restore()

# Check package status
renv::status()
```

3. **What gets created**:
- `renv.lock`: Lists exact package versions
- `renv/` folder: Local package library
- `.Rprofile`: Activates renv automatically

### Example renv.lock snippet:
```json
{
  "dplyr": {
    "Package": "dplyr",
    "Version": "1.1.3",
    "Source": "Repository",
    "Repository": "CRAN"
  }
}
```

## 🧪 2. Automated Testing

### Why Test Statistical Code?
Statistical analysis code is especially prone to errors:
- Data changes can break assumptions
- Small bugs can invalidate entire analyses
- Results need to be reproducible

### Types of Tests for Statistical Projects:

1. **Data Integrity Tests**:
   - Check data structure (rows, columns, types)
   - Verify no missing values where unexpected
   - Ensure data ranges are reasonable

2. **Model Validation Tests**:
   - Check that models converge
   - Verify expected coefficient signs
   - Test model assumptions

3. **Reproducibility Tests**:
   - Ensure same inputs produce same outputs
   - Check that random seeds work correctly

4. **Output Tests**:
   - Verify files are created where expected
   - Check plot generation
   - Validate saved model objects

### How to Run Tests:

1. **Locally**:
```r
# Run all tests
source("homework/01_poisson_logistic_regression/run_tests.R")

# Or manually
library(testthat)
test_dir("homework/01_poisson_logistic_regression/code/tests")
```

2. **What happens**:
- Tests check your data and models
- Detailed reports show what passed/failed
- Specific error messages help debugging

### Example Test Output:
```
✓ Dataset has correct structure
✓ Statistical summaries are reasonable  
✓ Poisson models fit correctly
✗ Output files are created correctly
  Error: modelo_poisson.rds not found
```

## 🔄 3. GitHub Actions (CI/CD)

### What is Continuous Integration?
Automatically run tests and build processes when you push code to GitHub.

### For Academic Projects, CI/CD can:
- **LaTeX**: Compile PDFs automatically
- **R Analysis**: Run statistical scripts
- **Quality Checks**: Validate code style
- **Testing**: Run automated tests

### Two Workflows Created:

#### A. LaTeX Compilation (`latex-compile.yml`)
**Triggers**: When you push changes to `report/` folders
**What it does**:
1. Installs LaTeX on a Ubuntu server
2. Finds all `main.tex` files
3. Compiles them to PDF (handles bibliography too)
4. Uploads PDFs as downloadable artifacts
5. Runs quality checks with `chktex`

**Benefits**:
- Never worry about LaTeX installation differences
- Get PDFs without running LaTeX locally
- Catch LaTeX errors early
- Share PDFs with collaborators easily

#### B. R Analysis (`r-analysis.yml`)
**Triggers**: When you push R code changes
**What it does**:
1. Sets up R environment
2. Installs your packages
3. Runs all R scripts in order
4. Executes automated tests
5. Generates analysis summary
6. Uploads results as artifacts

**Benefits**:
- Verify analysis works on clean systems
- Catch R errors before submission
- Document what outputs are generated
- Ensure reproducibility

### How to Use GitHub Actions:

1. **Push the workflow files** to your GitHub repository
2. **Make changes** to your LaTeX or R files
3. **Push changes** - workflows run automatically
4. **Check results** in the "Actions" tab on GitHub
5. **Download artifacts** (PDFs, analysis results)

### GitHub Interface:
```
Your Repository
├── Code tab
├── Issues tab  
├── Pull requests tab
└── Actions tab ← Workflows appear here
    ├── ✅ LaTeX Compilation (main) 
    ├── ✅ R Statistical Analysis (main)
    └── ❌ Failed workflow (shows errors)
```

### Workflow Status Badges:
Add to your README:
```markdown
![LaTeX](https://github.com/username/repo/workflows/Compile%20LaTeX%20Documents/badge.svg)
![R Analysis](https://github.com/username/repo/workflows/R%20Statistical%20Analysis/badge.svg)
```

## 🚀 Implementation Strategy

### Start Simple:
1. **Week 1**: Set up `renv` for current homework
2. **Week 2**: Add basic tests for your data
3. **Week 3**: Set up LaTeX compilation workflow
4. **Week 4**: Add R analysis workflow

### Gradual Adoption:
- Begin with one homework assignment
- Add more tests as you learn
- Expand workflows for future assignments
- Use lessons learned to improve setup

## 📋 Checklist for Full Implementation

### Package Management:
- [ ] Run `setup_renv.R` in homework directory
- [ ] Commit `renv.lock` and `.Rprofile` to git
- [ ] Update `.gitignore` to include `renv/library/`
- [ ] Test `renv::restore()` on different machine

### Testing:
- [ ] Create `code/tests/` directory
- [ ] Write basic data integrity tests
- [ ] Add model validation tests
- [ ] Set up `run_tests.R` script
- [ ] Run tests locally before pushing

### CI/CD:
- [ ] Push workflow files to `.github/workflows/`
- [ ] Test LaTeX compilation workflow
- [ ] Test R analysis workflow  
- [ ] Add status badges to README
- [ ] Set up notifications for failed builds

## 🔧 Troubleshooting

### Common Issues:

**renv problems**:
```r
# Reset renv if corrupted
renv::deactivate()
unlink("renv", recursive = TRUE)
renv::init()
```

**GitHub Actions fails**:
- Check the "Actions" tab for error logs
- Common issue: LaTeX packages missing
- Solution: Add missing packages to workflow

**Tests fail locally**:
- Check file paths are correct
- Ensure data files exist
- Verify package installations

## 📚 Further Reading

- [renv documentation](https://rstudio.github.io/renv/)
- [testthat package guide](https://testthat.r-lib.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [R packages book](https://r-pkgs.org/) (advanced)

## 💡 Benefits Summary

### For Academic Work:
- **Professors**: Can easily reproduce your results
- **Collaborators**: Get identical environments
- **Future you**: Recreate old analyses easily
- **Peer review**: Demonstrates reproducibility

### For Professional Development:
- **Industry skills**: CI/CD is standard in tech
- **Best practices**: Learn modern development workflows
- **Portfolio**: Shows advanced technical skills
- **Problem solving**: Automated testing catches issues early

This setup transforms your repository from a simple file collection into a modern, professional development environment that ensures reproducibility and catches errors automatically.
