Creating class instances in ``cuda-bindings`` should now be faster in most cases, because it
requires 1 heap allocation rather than 2. (#2821)
