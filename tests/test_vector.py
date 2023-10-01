"""Test vector module."""


from neat.vector import Vector2


def test_str() -> None:
    assert str(Vector2(3, 6)) == "(3, 6)"


def test_repr() -> None:
    assert repr(Vector2(3, 6)) == "Vector2(3, 6)"


def test_eq_vec() -> None:
    assert Vector2(3, 6) == Vector2(3, 6)


def test_eq_tuple() -> None:
    assert Vector2(3, 6) == (3, 6)


def test_init() -> None:
    assert Vector2((3, 82)) == Vector2(3, 82)


def test_from_points() -> None:
    assert Vector2.from_points((0, 3), (2, 5)) == Vector2(2, 2)


def test_get_magnitude() -> None:
    assert Vector2(3, 4).get_magnitude() == 5


def test_get_distance_to() -> None:
    assert Vector2(0, 0).get_distance_to((3, 4)) == 5


def test_copy() -> None:
    one = Vector2(5, 8)
    assert id(one.copy()) != id(one)


def test_normalize() -> None:
    vec = Vector2(3, 4)
    vec.normalize()
    assert vec == Vector2(3 / 5, 4 / 5)


def test_get_normalized() -> None:
    assert Vector2(3, 4).get_normalized() == Vector2(3 / 5, 4 / 5)


def test_add() -> None:
    assert Vector2(3, 4) + Vector2(5, 6) == Vector2(8, 10)


def test_sub() -> None:
    assert Vector2(3, 4) - Vector2(5, 6) == Vector2(-2, -2)


def test_neg() -> None:
    assert -Vector2(3, 4) == Vector2(-3, -4)


def test_mul() -> None:
    assert Vector2(5, 10) * 3 == Vector2(15, 30)


def test_truediv() -> None:
    assert Vector2(10, 5) / 2 == Vector2(5, 2.5)


def test_truediv_zero() -> None:
    assert Vector2(10, 5) / 0 == Vector2(10, 5)


def test_len() -> None:
    assert len(Vector2(5, 20)) == 2


def test_getitem() -> None:
    vec = Vector2(7, 28)
    assert vec[0] == 7
    assert vec[1] == 28
