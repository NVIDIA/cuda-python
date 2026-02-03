// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#include <Python.h>
#include <structmember.h>

/*
 * State for the module, storing references to our heap types.
 */
typedef struct {
    PyObject *FastEnum_Type;
    PyObject *FastEnumMetaclass_Type;
} fast_enum_state;

static inline fast_enum_state*
get_fast_enum_state(PyObject *module)
{
    void *state = PyModule_GetState(module);
    assert(state != NULL);
    return (fast_enum_state *)state;
}

/*
 * ===================================================================================
 *  _FastEnum Implementation (Inherits from int)
 * ===================================================================================
 */

typedef struct {
    PyLongObject long_base; /* Must be first to inherit from int */
    PyObject *name;
} FastEnumObject;

static PyObject *
FastEnum_get_singletons(PyObject *obj)
{
    return PyObject_GetAttrString(obj, "__singletons__");
}

static PyObject *
FastEnum_get_members(PyObject *obj)
{
    return PyObject_GetAttrString(obj, "__members__");
}

static void
FastEnum_dealloc(FastEnumObject *self)
{
    Py_CLEAR(self->name);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// This method is not exposed to Python and is only called from FastEnumMetaclass_init.
// It is used to make the initial singleton instances of each of the values.

static PyObject *
_FastEnum_new_member(PyTypeObject *type, PyObject *name, PyObject *value)
{
   /* Create the int base */
    PyObject *args_value = PyTuple_Pack(1, value);
    if (!args_value) return NULL;

    /* Call super().__new__(cls, value) - technically int.__new__ */
    PyObject *self_obj = PyLong_Type.tp_new(type, args_value, NULL);
    Py_DECREF(args_value);
    if (!self_obj) return NULL;

    FastEnumObject *self = (FastEnumObject *)self_obj;
    Py_INCREF(name);
    self->name = name;

    return self_obj;
}

// This method is exposed to Python as a constructor, and always returns one of
// the singleton instances.

static PyObject *
FastEnum_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds) != 0) {
        PyErr_Format(PyExc_TypeError, "%N does not take keyword arguments", type);
        return NULL;
    }

    if (PyTuple_Size(args) != 1) {
        PyErr_Format(PyExc_ValueError, "%N takes exactly one argument", type);
        return NULL;
    }

    PyObject *val = PyTuple_GET_ITEM(args, 0);
    PyObject *singletons = FastEnum_get_singletons((PyObject *)type);

    int contains = PyDict_Contains(singletons, val);

    switch (contains) {
        case 1:
            PyObject *result = PyDict_GetItem(singletons, val); // borrowed ref
            Py_DECREF(singletons);
            Py_INCREF(result);
            return result;
        case 0:
            Py_DECREF(singletons);
            PyErr_Format(PyExc_ValueError, "Value %S not in %S", val, type);
            return NULL;
        case -1:
            Py_DECREF(singletons);
            return NULL;
    }
}

static PyObject *
FastEnum_repr(FastEnumObject *self)
{
    PyObject *result = NULL;
    PyObject *type = NULL;
    PyObject *type_name = NULL;
    long long_val;

    /* Equivalent to: f"<{type(self).__name__}.{self._name}: {int(self)}>" */

    type = (PyObject *)PyObject_Type((PyObject *)self);
    if (type == NULL) goto exit;

    type_name = PyObject_GetAttrString(type, "__name__");
    if (type_name == NULL) goto exit;

    long_val = PyLong_AsLong((PyObject *)self);

    result = PyUnicode_FromFormat("<%U.%U: %ld>", type_name, self->name, long_val);

  exit:
    Py_XDECREF(type);
    Py_XDECREF(type_name);

    return result;
}

static PyObject *
FastEnum_get_value(FastEnumObject *self, void *closure)
{
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
FastEnum_get_name(FastEnumObject *self, void *closure)
{
    Py_INCREF(self->name);
    return self->name;
}

static PyGetSetDef FastEnum_getseters[] = {
    {"value", (getter)FastEnum_get_value, NULL, "Enum value", NULL},
    {"name", (getter)FastEnum_get_name, NULL, "Enum name", NULL},
    {NULL}  /* Sentinel */
};

static PyType_Slot FastEnum_slots[] = {
    {Py_tp_new, FastEnum_new},
    {Py_tp_dealloc, FastEnum_dealloc},
    {Py_tp_repr, FastEnum_repr},
    {Py_tp_getset, FastEnum_getseters},
    {0, NULL},
};

static PyType_Spec FastEnum_spec = {
    .name = "_fast_enum._FastEnum",
    .basicsize = sizeof(FastEnumObject),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_LONG_SUBCLASS,
    .slots = FastEnum_slots,
};

/*
 * ===================================================================================
 *  _FastEnumMetaclass Implementation (Inherits from type)
 * ===================================================================================
 */

/* Since it inherits from type, the struct is technically PyHeapTypeObject
   but we don't need extra fields based on the python code. */

static int
FastEnumMetaclass_init(PyObject *cls, PyObject *args, PyObject *kwds)
{
    PyObject *members = NULL;
    PyObject *singletons = NULL;
    PyObject *contents = NULL;
    PyObject *dunder = NULL;
    int result = -1;

    if (PyType_Type.tp_init(cls, args, kwds) < 0) {
        return -1;
    }

    contents = PyObject_GenericGetDict(cls, NULL);
    if (contents == NULL) goto exit;

    members = PyDict_New();
    if (members == NULL) goto exit;

    singletons = PyDict_New();
    if (singletons == NULL) goto exit;

    dunder = PyUnicode_FromString("__");
    if (dunder == NULL) goto exit;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(contents, &pos, &key, &value)) {
        // Don't convert __dunder__ members
        if (
            PyUnicode_Tailmatch(key, dunder, 0, 2, 0) &&
            PyUnicode_Tailmatch(key, dunder, 0, 2, -1)
        ) {
            continue;
        }

        // Only convert members with integer values
        if (PyLong_Check(value)) {
            PyObject *new_entry = _FastEnum_new_member((PyTypeObject *)cls, key, value);
            if (new_entry == NULL) goto exit;
            if (PyDict_SetItem(members, key, new_entry)) {
                Py_DECREF(new_entry);
                goto exit;
            }

            if (PyDict_SetItem(singletons, value, new_entry)) {
                Py_DECREF(new_entry);
                goto exit;
            }

            Py_DECREF(new_entry);
        }
    }

    if (PyObject_SetAttrString(cls, "__members__", members)) goto exit;
    if (PyObject_SetAttrString(cls, "__singletons__", singletons)) goto exit;

    pos = 0;
    while (PyDict_Next(members, &pos, &key, &value)) {
        if (PyObject_SetAttr(cls, key, value)) {
            goto exit;
        }
    }

    result = 0;

  exit:
    Py_XDECREF(members);
    Py_XDECREF(singletons);
    Py_XDECREF(contents);
    Py_XDECREF(dunder);
    return result;
}

static PyObject *
FastEnumMetaclass_repr(PyObject *self)
{
    PyObject *result = NULL;
    PyObject *type_name = NULL;

    /* Equivalent to: f"<enum '{self.__name__}'>" */

    type_name = PyObject_GetAttrString(self, "__name__");
    if (type_name == NULL) return NULL;

    result = PyUnicode_FromFormat("<enum '%U'>", type_name);
    Py_DECREF(type_name);

    return result;
}

static Py_ssize_t
FastEnumMetaclass_len(PyObject *cls)
{
    PyObject *members = FastEnum_get_members(cls);
    if (!members) return -1;
    Py_ssize_t len = PyDict_Size(members);
    Py_DECREF(members);
    return len;
}

static PyObject *
FastEnumMetaclass_iter(PyObject *cls)
{
    PyObject *members = FastEnum_get_members(cls);
    if (!members) return NULL;

    PyObject *values = PyDict_Values(members);
    Py_DECREF(members);
    if (!values) return NULL;

    PyObject *iter = PyObject_GetIter(values);
    Py_DECREF(values);
    return iter;
}

static int
FastEnumMetaclass_sq_contains(PyObject *cls, PyObject *value)
{
    PyObject *singletons = FastEnum_get_singletons(cls);
    if (!singletons) return -1;

    int in_keys = PyDict_Contains(singletons, value);

    Py_DECREF(singletons);
    return in_keys;
}

static PyType_Slot FastEnumMetaclass_slots[] = {
    {Py_tp_init, FastEnumMetaclass_init},
    {Py_tp_repr, FastEnumMetaclass_repr},
    {Py_sq_length, FastEnumMetaclass_len},
    {Py_tp_iter, FastEnumMetaclass_iter},
    {Py_sq_contains, FastEnumMetaclass_sq_contains},
    {Py_tp_base, NULL}, /* Will be &PyType_Type */
    {0, NULL},
};

static PyType_Spec FastEnumMetaclass_spec = {
    .name = "_fast_enum._FastEnumMetaclass",
    .basicsize = sizeof(PyHeapTypeObject), /* Metaclasses must use PyHeapTypeObject size */
    .itemsize = sizeof(PyMemberDef),       /* Inheriting from type requires itemsize */
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_TYPE_SUBCLASS,
    .slots = FastEnumMetaclass_slots,
};

/*
 * ===================================================================================
 *  Module Initialization (Multi-phase)
 * ===================================================================================
 */

static int
fast_enum_exec(PyObject *module)
{
    fast_enum_state *state = get_fast_enum_state(module);

    /* Initialize _FastEnum (bases: int) */
    state->FastEnum_Type = PyType_FromModuleAndSpec(module, &FastEnum_spec, (PyObject *)&PyLong_Type);
    if (state->FastEnum_Type == NULL) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "_FastEnum", state->FastEnum_Type) < 0) {
        Py_DECREF(state->FastEnum_Type);
        return -1;
    }

    /* Initialize _FastEnumMetaclass (bases: type) */
    state->FastEnumMetaclass_Type = PyType_FromModuleAndSpec(module, &FastEnumMetaclass_spec, (PyObject *)&PyType_Type);
    if (state->FastEnumMetaclass_Type == NULL) {
        return -1;
    }
    if (PyModule_AddObjectRef(module, "_FastEnumMetaclass", state->FastEnumMetaclass_Type) < 0) {
         Py_DECREF(state->FastEnumMetaclass_Type);
         return -1;
    }

    return 0;
}

static int
fast_enum_traverse(PyObject *module, visitproc visit, void *arg)
{
    fast_enum_state *state = get_fast_enum_state(module);
    Py_VISIT(state->FastEnum_Type);
    Py_VISIT(state->FastEnumMetaclass_Type);
    return 0;
}

static int
fast_enum_clear(PyObject *module)
{
    fast_enum_state *state = get_fast_enum_state(module);
    Py_CLEAR(state->FastEnum_Type);
    Py_CLEAR(state->FastEnumMetaclass_Type);
    return 0;
}

static PyModuleDef_Slot fast_enum_slots[] = {
    {Py_mod_exec, fast_enum_exec},
    {0, NULL}
};

static struct PyModuleDef _fast_enum_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_fast_enum",
    .m_doc = "Fast Enum C Extension",
    .m_size = sizeof(fast_enum_state),
    .m_slots = fast_enum_slots,
    .m_traverse = fast_enum_traverse,
    .m_clear = fast_enum_clear,
};

PyMODINIT_FUNC
PyInit__fast_enum(void)
{
    return PyModuleDef_Init(&_fast_enum_module);
}
