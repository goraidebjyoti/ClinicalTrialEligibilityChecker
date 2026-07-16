Assets Directory
================

This directory contains static assets used by the
ClinicalTrialEligibilityChecker client application.

Assets in this folder are loaded by the Streamlit-based user interface
and are not involved in any model inference or data processing logic.


CONTENTS
--------

logo.png
--------
- Official logo used in the client user interface.
- Displayed in the application header and browser tab icon.
- Used to visually identify the ClinicalTrialEligibilityChecker system.


USAGE
-----
- Assets are referenced by relative path from app.py.
- The logo is rendered in the header using Streamlit's image component.
- The same image is used as the application page icon.


DESIGN NOTES
------------
- The logo is intentionally lightweight to ensure fast UI rendering.
- Any changes to dimensions should preserve clarity at small sizes.
- Aspect ratio should remain unchanged to avoid distortion.


MODIFICATION GUIDELINES
-----------------------
- Do not rename asset files without updating references in app.py.
- Replace assets only with approved versions to maintain consistency.
- Additional assets (icons, diagrams) should be documented here
  if introduced in the future.


SCOPE
-----
- This directory contains only presentation-related resources.
- No patient data, trial data, or model outputs are stored here.


LICENSE
-------
Internal research and demonstration use only.
Not for redistribution without permission.
