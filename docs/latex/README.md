# Visual midterm report

`midterm-report.tex` is the main editable report; `iterations.tex` holds the Baseline and Iteration results sections (Iterations 2–5, plus selection and review). `midterm_report.pdf` is the compiled copy. All required images are included in `figures/`, so this folder can be uploaded to Overleaf with `midterm-report.tex` as the main document.

From the repository root:

```powershell
python scripts/build_report_figures.py
cd docs/latex
tectonic midterm-report.tex
```

`pdflatex midterm-report.tex` also works with the listed standard LaTeX packages installed.

The charts read the checked-in result JSON. The photo panel displays an unchanged row of `docs/images/prompt-ablation.jpg`, cropped by LaTeX to frame 30. It is an initialization comparison, not a tracking result. No experiment was rerun for this rewrite.

The report separates original code, project additions, and pretrained model integrations. Proposed next checks are plans, not completed experiments. Experiment dates stay October 11; the draft revision is November 19.
