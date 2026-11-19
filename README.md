# Text Filtering and Ranking for Security Bug Report Prediction

This project replicates a study that uses text filtering and ranking to predict which bug reports describe security issues.

## Implementation status

The filtering stage is implemented. In a test with 1,000 bug reports (BRs), about 50 were security bug reports (SBRs). After filtering, 220 non-security bug reports (NSBRs) remained.

The filter appeared to be more selective than the one reported in the original paper. One possible explanation is that the original method did not use stemming, so its list of security-related keywords may have included multiple forms of the same word. Whether this stricter filtering improves prediction performance remains to be evaluated.
