:attr:`LegacyPinnedMemoryResource.device_id` now returns ``-1``, as
documented for memory that is not bound to a device and as
:class:`PinnedMemoryResource` already does, instead of raising
``RuntimeError``. :attr:`Buffer.device_id` on such a buffer returns ``-1``
as well, which also lets a pinned buffer back a linear or pitched texture
resource. (#2750)
