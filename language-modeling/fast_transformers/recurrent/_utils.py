import warnings


def check_state(state=None, memory=None):
    if memory is not None:
        warnings.warn(("'memory' is deprecated for recurrent transformers "
                       " and will be removed in the future, use 'state' "
                       "instead"), DeprecationWarning)
    if state is None:
        state = memory
    return state
