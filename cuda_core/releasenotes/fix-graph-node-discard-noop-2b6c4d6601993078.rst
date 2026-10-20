Calling ``discard()`` on the set-like :attr:`graph.GraphNode.pred` and
:attr:`graph.GraphNode.succ` views with an object that is not a
:class:`~graph.GraphNode` is now a no-op instead of raising ``TypeError``,
consistent with :class:`collections.abc.MutableSet` semantics. (#2893)
