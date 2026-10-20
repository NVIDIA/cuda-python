:meth:`Device.set_current` with an explicit :class:`Context` now switches
contexts with a single driver call, so a failure leaves the previous context
current instead of leaving the thread with no context. It also works when no
context is current, returning ``None``. (#2759)
