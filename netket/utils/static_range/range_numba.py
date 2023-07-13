# Copyright 2022 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##########################################################################
##                                                                      ##
## This file provides the Numba implementation of the `Range` object    ##
## defined in range.py. The file uses the low-level implementation      ##
## API of numba to define it.                                           ##
##                                                                      ##
## The reason why @jitclass is not used is because (i) it has been      ##
## deprecated and (ii) we need two different objects in python and      ##
## numba, because the python one should play well with Jax.             ##
##                                                                      ##
##########################################################################


import numpy as np
import operator

import numba
from numba import types
from numba.core import cgutils
from numba.extending import (
    models,
    register_model,
    lower_builtin,
    typeof_impl,
    NativeValue,
    make_attribute_wrapper,
)

from .range import Range


# Define Numba type
class RangeType(types.Type):
    def __init__(self, startT, stepT):
        self.startT = startT
        self.stepT = stepT
        super(RangeType, self).__init__(name=f"Range[{startT}, {stepT}]")

    def __class_getitem__(clz, args):
        if not isinstance(args, tuple):
            args = (args,)
        return clz(*args)


@typeof_impl.register(Range)
def typeof_index(val, c):
    return RangeType(numba.typeof(val.start), numba.typeof(val.step))


# Define Numba type implementation
@register_model(RangeType)
class RangeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("start", fe_type.startT),
            ("step", fe_type.stepT),
            ("length", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


# Allow those fields to be accessed.
make_attribute_wrapper(RangeType, "start", "start")
make_attribute_wrapper(RangeType, "step", "step")
make_attribute_wrapper(RangeType, "length", "length")

# Box/Unbox is the Python <-> Numba conversion routines to convert
# the python object to the numba one.


@numba.extending.unbox(RangeType)
def unbox_interval(typ, obj, c):
    """
    Convert a Interval object to a native interval structure.
    """
    start_obj = c.pyapi.object_getattr_string(obj, "start")
    step_obj = c.pyapi.object_getattr_string(obj, "step")
    len_obj = c.pyapi.object_getattr_string(obj, "length")

    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    interval.start = c.pyapi.to_native_value(typ.startT, start_obj).value
    interval.step = c.pyapi.to_native_value(typ.stepT, step_obj).value
    interval.length = c.pyapi.long_as_longlong(len_obj)
    c.pyapi.decref(start_obj)
    c.pyapi.decref(step_obj)
    c.pyapi.decref(len_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(interval._getvalue(), is_error=is_error)


@numba.extending.box(RangeType)
def box_interval(typ, val, c):
    """
    Convert a native interval structure to an Interval object.
    """
    interval = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    start_obj = c.pyapi.from_native_value(typ.startT, interval.start)
    step_obj = c.pyapi.from_native_value(typ.stepT, interval.step)
    len_obj = c.pyapi.long_from_longlong(interval.length)

    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Range))
    res = c.pyapi.call_function_objargs(class_obj, (start_obj, step_obj, len_obj))
    c.pyapi.decref(start_obj)
    c.pyapi.decref(step_obj)
    c.pyapi.decref(len_obj)
    return res


## Numba implementations


# Define the Range constructor that can be called within Numba code
# This is probably unneded but let's keep it around.
# This is the type inference phase...
# @type_callable(Range)
# def type_interval(context):
#     def typer(start, step, length):
#         if (
#             isinstance(start, (types.Float, types.Integer))
#             and isinstance(step, (types.Float, types.Integer))
#             and isinstance(length, (types.Integer,))
#         ):
#             return range_type
#
#     return typer


# this is the implementation
@lower_builtin(Range, types.Float, types.Float, types.Integer)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    start, step, length = args
    ran = cgutils.create_struct_proxy(typ)(context, builder)
    ran.start = start
    ran.step = step
    ran.length = length

    return ran._getvalue()


## Implements methods for numba


@numba.extending.overload(len)
def tuple_len(r):
    if isinstance(r, RangeType):

        def len_impl(r):
            return r.length

        return len_impl


@numba.extending.overload(list)
@numba.extending.overload(np.array)
@numba.extending.overload(np.asarray)
def range_asarray(r, dtype=None):
    if isinstance(r, RangeType):

        def range_asarray_impl(r, dtype=None):
            return r.start + np.arange(r.length + 1, dtype=dtype) * r.step

        return range_asarray_impl


@numba.extending.overload(operator.getitem)
def getitem_range(r, i):
    if isinstance(r, RangeType):
        if isinstance(i, types.Integer):

            def getitem_range_impl(r, i):
                if i > r.length:
                    raise IndexError
                return r.start + r.step * i

            return getitem_range_impl


@numba.extending.overload_method(RangeType, "find")
def range_find(ran, vals):
    if isinstance(vals, types.Array):

        def find_impl_arr(ran, vals):
            return np.asarray((vals - ran.start) / ran.step, dtype=np.int64)

        return find_impl_arr
    else:

        def find_impl(ran, vals):
            return int((vals - ran.start) / ran.step)

        return find_impl


@numba.extending.overload_method(RangeType, "basis_to_index")
def range_basis_to_index(ran, vals):
    if isinstance(vals, types.Array):

        def basis_to_index_impl_arr(ran, vals):
            return np.asarray((vals - ran.start) / ran.step, dtype=np.int32)

        return basis_to_index_impl_arr
    else:

        def basis_to_index_impl(ran, vals):
            return int((vals - ran.start) / ran.step)

        return basis_to_index_impl


@numba.extending.overload_method(RangeType, "index_to_basis")
def range_index_to_basis(ran, vals):
    def index_to_basis_impl(ran, vals):
        return ran.start + ran.step * vals

    return index_to_basis_impl


@numba.extending.overload_method(RangeType, "flip_state")
def range_flip_state(ran, vals):
    def range_flip_state_impl(ran, vals):
        if len(ran) != 2:
            raise NotImplementedError
        constant_sum = 2 * ran.start + ran.step
        return constant_sum - vals

    return range_flip_state_impl
