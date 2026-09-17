"""No unit tests yet for evaluate.py.

The module loads the test dataset and sketch_model.pt at import time, and its
only helper is a forward-hook callback.  Keep evaluation coverage at the
integration-test level until its metric/plotting logic is extracted into pure
functions that can be imported without those artifacts.
"""
