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


@_tensor_member_fn
@al.jit
def batchnorm2d(input:al.array,scale:al.array,offset:al.array,mean:al.array,variance:al.array)->al.array:
    "Compute the batchnrom of the input with mean,variance"
    """
    :params: input
    :params: scale
    :params: offset
    :params: mean
    :params: variance
    """
    return al.prim.batchnorm2d(input,scale,offset,mean,variance)

@_tensor_member_fn
@al.jit
def maxpool2d(input)->al.array:
    "Compute the maxpool2d of the input "
    return al.prim.maxpool2d(input)

@_tensor_member_fn
@al.jit
def conv2d(input,weight,window_stride, lhs_dilation, rhs_dilation,
        padding_args, window_reversal)->al.array:
    "Compute the maxpool2d of the input "
    return al.prim.conv2d(input,weight,window_stride, lhs_dilation, rhs_dilation,
        padding_args, window_reversal)


    