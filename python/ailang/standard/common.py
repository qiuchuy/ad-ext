import inspect
import ailang as al


def _tensor_member_fn(fn):
    """Decorator that adds this free function as a member fn on class tensor.

    When called as a member function on class tensor, the first argument to `fn`
    is `self`, i.e. the tensor object.

    If there are multiple decorators on a function, you probably want this one
    to be the highest one (i.e. furthest from the function's `def`), so it's
    applied last.

    Unfortunately you still need to add a type stub to the body of class tensor
    in order for pytype to know about it.
    """
    assert callable(fn)
    orig_sig = inspect.signature(fn)
    # Does fn take args other than _builder, _generator, and the tensor itself?
    has_args = len(orig_sig.parameters.keys() - {"_builder", "_generator"}) > 1

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f"""
    This function can also be called as a member function on :py:class:`array`,
    as :code:`x.{fn.__name__}({"..." if has_args else ""})` instead of
    :code:`{fn.__name__}(x{", ..." if has_args else ""})`.
    """

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Match the signature of `fn`, but change the first arg to `self` so the
    # docs are a little less weird.
    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name="self")
    new_sig = orig_sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"
    # If fn is a builtin, mark the wrapper as a builtin too.

    setattr(al.array, fn.__name__, wrapper)
    return fn


def flatten_pytree(pytree):
    """Flatten a pytree into a list."""
    leaves = []

    def _flatten_pytree(pytree):
        if isinstance(pytree, (tuple, list)):
            for x in pytree:
                _flatten_pytree(x)
        else:
            leaves.append(pytree)

    _flatten_pytree(pytree)
    return leaves


def element_wise(f):
    r"""
    Decorator for element-wise binary operations on tensors.

    This decorator automates the process of applying element-wise binary operations
    to tensors, handling shape broadcasting automatically. It ensures that the
    decorated function receives tensors of compatible shapes for element-wise
    operations.

    Parameters:
    f : callable
        The binary function to be decorated. It should take two tensors as input
        and return a single tensor.

    Returns:
    callable
        A wrapped function that handles tensor broadcasting before applying the
        original function.

    Behavior:
    1. Checks if exactly two tensor arguments are provided.
    2. Determines the broadcasted shape based on the input tensors' shapes.
    3. Expands both input tensors to the broadcasted shape.
    4. Applies the original function to the expanded tensors.

    Notes:
    ------
    - The decorated function should expect two tensor arguments.
    - Broadcasting follows NumPy-style rules: dimensions are aligned from
      right to left, and dimensions of size 1 are stretched to match the
      other tensor's size in that dimension.
    - Additional keyword arguments are passed through to the original function.

    Raises:
    -------
    ValueError
        If the number of positional arguments is not exactly two.

    Example:
    --------
    @element_wise_binary
    def add_tensors(t1, t2):
        return t1 + t2

    # Now add_tensors can handle tensors of different shapes, e.g.:
    # result = add_tensors(tensor([1, 2, 3]), tensor([[1], [2]]))
    """

    def broadcast_info(shape1, shape2):
        # Reverse shapes for easy handling
        shape1 = shape1[::-1]
        shape2 = shape2[::-1]

        # Determine the maximum length of the shapes
        max_len = max(len(shape1), len(shape2))

        # Pad the shorter shape with ones
        shape1.extend([1] * (max_len - len(shape1)))
        shape2.extend([1] * (max_len - len(shape2)))

        # Calculate the broadcasted shape
        broadcasted_shape = []
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == dim2 or dim1 == 1:
                broadcasted_shape.append(dim2)
            elif dim2 == 1:
                broadcasted_shape.append(dim1)
            else:
                raise ValueError("Incompatible shapes for broadcasting.")

        # Reverse the broadcasted shape to the correct order
        broadcasted_shape.reverse()

        # Determine which array needs broadcasting
        array1_needs_broadcasting = (shape1[::-1]) != broadcasted_shape
        array2_needs_broadcasting = (shape2[::-1]) != broadcasted_shape
        return broadcasted_shape, array1_needs_broadcasting, array2_needs_broadcasting

    def wrapper(*args, **kwargs):
        # Ensure we have exactly two tensors
        if len(args) != 2:
            raise ValueError(
                "element_wise_binary decorator expects exactly two tensor arguments."
            )
        array1, array2 = args
        if not isinstance(array1, al.tracer):
            array1 = al.create_tracer(array1)
        else:
            array1 = al.promote_tracer(array1)
        if not isinstance(array2, al.tracer):
            array2 = al.create_tracer(array2)
        else:
            array2 = al.promote_tracer(array2)
        shape1 = list(array1.shape)
        shape2 = list(array2.shape)
        broadcasted_shape, array1_need_broadcast, array2_need_broadcast = (
            broadcast_info(shape1, shape2)
        )
        expanded_array1 = array1
        expanded_array2 = array2
        if array1_need_broadcast:
            expanded_array1 = al.prim.broadcast_to(array1, broadcasted_shape)
        if array2_need_broadcast:
            expanded_array2 = al.prim.broadcast_to(array2, broadcasted_shape)
        result = f(expanded_array1, expanded_array2, **kwargs)
        return result

    return wrapper
