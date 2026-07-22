"""Streamlit Community Cloud entrypoint for MAGI.

The application is implemented in ``module3_decision_report``. This root file
exists because Streamlit Community Cloud detects ``streamlit_app.py`` by
default.
"""

from decision_report.app import main

main()
