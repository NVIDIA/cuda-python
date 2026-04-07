# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from sphinx.ext.autodoc import ClassDocumenter


def _sanitize_enum_member_doc(doc: str) -> str:
    return doc.replace("`", "``").replace("*", r"\*")


class EnumDocumenter(ClassDocumenter):
    objtype = "enum"
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)  # noqa

    @classmethod
    def can_document_member(cls, member, _membername, _isattr, _parent):
        return hasattr(member, "__members__")

    def get_doc(self):
        docs = super().get_doc()

        if (
            docs
            and self.object.__module__ == "cuda.bindings.nvml"
            and self.object.__name__ == "GpmMetricId"
        ):
            return [["GPM Metric Identifiers.", "", "See ``nvmlGpmMetricId_t``."]]

        return docs

    def add_content(self, more_content):
        super().add_content(more_content)

        source_name = self.get_sourcename()
        enum_object = self.object
        if not enum_object.__doc__:
            self.add_line(enum_object.__name__, source_name)
        self.add_line("", source_name)

        for member_name, enum_member in enum_object.__members__.items():  # type: ignore[attr-defined]
            member_value = enum_member.value

            self.add_line(f"**{member_name}**: {member_value}", source_name)
            if enum_member.__doc__:
                member_doc = enum_member.__doc__
                if enum_object.__module__ == "cuda.bindings.nvml" and enum_object.__name__ == "GpmMetricId":
                    member_doc = _sanitize_enum_member_doc(member_doc)
                self.add_line(f"    {member_doc}", source_name)
            self.add_line("", source_name)


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension
    app.add_autodocumenter(EnumDocumenter)
    return {
        "version": "1",
        "parallel_read_safe": True,
    }
