# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


from sphinx.ext.autodoc import ClassDocumenter, Documenter


class EnumDocumenter(ClassDocumenter):
    objtype = 'enum'
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Documenter
    ) -> bool:
        return hasattr(member, "__members__")

    def add_content(
        self,
        more_content: StringList | None,
    ) -> None:
        super().add_content(more_content)

        source_name = self.get_sourcename()
        enum_object = self.object
        if not enum_object.__doc__:
            self.add_line(enum_object.__name__, source_name)
        self.add_line('', source_name)

        for member_name, enum_member in enum_object.__members__.items():  # type: ignore[attr-defined]
            member_value = enum_member.value

            self.add_line(f'**{member_name}**: {member_value}', source_name)
            if enum_member.__doc__:
                self.add_line(f'    {enum_member.__doc__}', source_name)
            self.add_line('', source_name)


def setup(app):
    app.setup_extension('sphinx.ext.autodoc')  # Require autodoc extension
    app.add_autodocumenter(EnumDocumenter)
    return {
        'version': '1',
        'parallel_read_safe': True,
    }