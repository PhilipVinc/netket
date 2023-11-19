from typing import Generic, TypeVar

import jax
import pytest
from flax import serialization

from netket.utils.struct import Pytree, field, static_field


class TestPytree:
    def test_attribute_error(self):
        class Foo(Pytree):
            x: int = static_field()

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        with pytest.raises(AttributeError):
            Foo(y=3)

        class Foo(Pytree, dynamic_nodes=True):
            x: int = static_field()

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        Foo(y=3)

    def test_imm(self):
        class Foo(Pytree):
            x: int = static_field()
            y: int

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(
            AttributeError, match="is immutable, trying to update field"
        ):
            pytree.x = 4

    def test_jit(self):
        class Foo(Pytree):
            a: int
            b: int = static_field()

            def __init__(self, a, b):
                self.a = a
                self.b = b

        module = Foo(a=1, b=2)

        @jax.jit
        def f(m: Foo):
            return m.a + m.b

        assert f(module) == 3

    def test_flax_serialization(self):
        class Bar(Pytree, dynamic_nodes=True):
            a: int = static_field()

            def __init__(self, a, b):
                self.a = a
                self.b = b

        class Foo(Pytree):
            bar: Bar
            c: int
            d: int = static_field()

            def __init__(self, bar, c, d):
                self.bar = bar
                self.c = c
                self.d = d

        foo: Foo = Foo(bar=Bar(a=1, b=2), c=3, d=4)

        state_dict = serialization.to_state_dict(foo)

        assert state_dict == {
            "bar": {
                "b": 2,
            },
            "c": 3,
        }

        state_dict["bar"]["b"] = 5

        foo = serialization.from_state_dict(foo, state_dict)

        assert foo.bar.b == 5

        del state_dict["bar"]["b"]

        with pytest.raises(ValueError, match="Missing field"):
            serialization.from_state_dict(foo, state_dict)

        state_dict["bar"]["b"] = 5

        # add unknown field
        state_dict["x"] = 6

        with pytest.raises(ValueError, match="Unknown field"):
            serialization.from_state_dict(foo, state_dict)

    def test_generics(self):
        T = TypeVar("T")

        class MyClass(Pytree, Generic[T]):
            def __init__(self, x: T):
                self.x = x

        MyClass[int]

    def test_key_paths(self):
        class Bar(Pytree):
            a: int = 1
            b: int = static_field(default=2)

            def __init__(self, a=1, b=2):
                self.a = a
                self.b = b

        class Foo(Pytree):
            x: int = 2
            y: int = static_field(default=1)
            z: Bar = field(default_factory=Bar)

            def __init__(self, x=3, y=4, z=None):
                self.x = x
                self.y = y
                if z is None:
                    z = Bar()
                self.z = z

        foo = Foo()

        path_values, treedef = jax.tree_util.tree_flatten_with_path(foo)
        path_values = [(list(map(str, path)), value) for path, value in path_values]

        assert path_values[0] == ([".x"], 3)
        assert path_values[1] == ([".z", ".a"], 1)

    def test_setter_attribute_allowed(self):
        n = None

        class SetterDescriptor:
            def __set__(self, _, value):
                nonlocal n
                n = value

        class Foo(Pytree):
            x = SetterDescriptor()

        foo = Foo()
        foo.x = 1

        assert n == 1

    def test_replace_unknown_fields_error(self):
        class Foo(Pytree):
            pass

        with pytest.raises(ValueError, match="Trying to replace unknown fields"):
            Foo().replace(y=1)

    def test_pytree_with_new(self):
        class A(Pytree):
            a: int

            def __init__(self, a):
                self.a = a

            def __new__(cls, a):
                return super().__new__(cls)

        pytree = A(a=1)

        pytree = jax.tree_map(lambda x: x * 2, pytree)

    @pytest.mark.parametrize("dynamic_nodes", [True, False])
    def test_deterministic_order(self, dynamic_nodes):
        class A(Pytree, dynamic_nodes=dynamic_nodes):
            if not dynamic_nodes:
                a: int
                b: int

            def __init__(self, order: bool):
                if order:
                    self.a = 1
                    self.b = 2
                else:
                    self.b = 2
                    self.a = 1

        p1 = A(order=True)
        p2 = A(order=False)

        leaves1 = jax.tree_util.tree_leaves(p1)
        leaves2 = jax.tree_util.tree_leaves(p2)

        assert leaves1 == leaves2


class TestMutablePytree:
    def test_pytree(self):
        class Foo(Pytree, mutable=True):
            x: int = static_field()
            y: int

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4

    @pytest.mark.parametrize("dynamic_nodes", [True, False])
    def test_no_new_fields_after_init(self, dynamic_nodes):
        class Foo(Pytree, mutable=True, dynamic_nodes=dynamic_nodes):
            x: int = static_field()

            def __init__(self, x):
                self.x = x

        foo = Foo(x=1)
        foo.x = 2

        with pytest.raises(AttributeError, match=r"Cannot add new fields to"):
            foo.y = 2