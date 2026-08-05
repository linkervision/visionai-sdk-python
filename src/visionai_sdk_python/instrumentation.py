"""Source attribution for outbound HTTP, with no call-site changes.

Call ``instrument()`` once at service startup::

    from visionai_sdk_python import instrumentation
    instrumentation.instrument()

Every request through ``requests``, ``httpx`` (sync and async) and ``aiohttp``
then carries ``X-Request-Source`` -- including third-party code, module-level
one-shots (``requests.get()``), and clients built before ``instrument()`` ran,
since injection patches a request-dispatch method on the class rather than
``__init__``.

**Destination scoping.** Only hosts matching ``allowed_destination_hosts``
(default covers ``vlm-inference-server``, ``vlm-scheduling-service``, and
``*.svc.cluster.local``) get the header; everything else (Stripe, an external
webhook, ...) is untouched. Fail-closed, no "all destinations" mode -- opt in
explicitly::

    instrumentation.instrument(
        allowed_destination_hosts=["vlm-inference-server", "*.svc.cluster.local"],
    )

Re-checked on every hop, so a redirect leaving the allowlist stops carrying the
header instead of leaking it onward.

**Forwarding an inherited origin (A-5).** By default this stamps the local
service's own identity. If service B calls VLM on behalf of service A, B
should forward A's identity instead -- use ``current_origin()``/
``origin_from_headers()`` from your inbound-request middleware::

    origin = instrumentation.origin_from_headers(incoming_request.headers)
    with instrumentation.current_origin(origin):
        ...  # any outbound call made in this scope inherits it

Injection runs per-request rather than at client construction, so this works
even through a client built once at startup and reused across many requests.
``source_headers()`` is an escape hatch for sending the header to a
destination outside ``allowed_destination_hosts`` on purpose -- returns
``{"X-Request-Source": value}`` or ``{}``. Don't pass ``get_current_origin()``
directly as a header value: it's ``None`` outside any scope, and ``httpx``
raises ``TypeError`` on a ``None``-valued header. See "A-5" in the
service-source-attribution plan.
"""

import contextvars
import fnmatch
import warnings
from typing import Any
from urllib.parse import urlsplit

try:
    import wrapt
except ImportError as e:
    raise ImportError(
        "visionai_sdk_python.instrumentation requires the 'wrapt' package. "
        "Install it with: pip install visionai-sdk-python[instrumentation]"
    ) from e

from ._source_header import (
    INJECTED_EXTENSION_KEY,
    SOURCE_HEADER,
    _effective_source,
    current_origin,
    get_current_origin,
    origin_from_headers,
    source_headers,
)

__all__ = [
    "instrument",
    "uninstrument",
    "current_origin",
    "get_current_origin",
    "origin_from_headers",
    "source_headers",
]

_instrumented = False

# Fail-closed: a service that calls instrument() with no argument only gets
# the header injected on its own cluster-internal backends. There is
# deliberately no "match everything" mode -- a service that needs a different
# set names it explicitly via allowed_destination_hosts.
#
# Named explicitly rather than relying on a single glob: "*-backend.svc.cluster.local"
# alone doesn't match the actual current VLM targets (vlm-inference-server,
# vlm-scheduling-service), which would make the out-of-the-box default silently
# inject nothing for the primary use case, with no signal that it didn't.
_DEFAULT_ALLOWED_DESTINATION_HOSTS: tuple[str, ...] = (
    "vlm-inference-server",
    "vlm-scheduling-service",
    "*-backend.svc.cluster.local",
    "*.svc.cluster.local",
)

_allowed_destination_hosts: tuple[str, ...] = _DEFAULT_ALLOWED_DESTINATION_HOSTS


def _extract_host(url: Any) -> str | None:
    """The hostname component of a URL, via a real URL parser.

    Handles ``requests``' plain string URLs, ``httpx.URL`` and ``yarl.URL``
    (both expose ``.host`` already parsed out) uniformly, rather than doing any
    string matching against the whole URL -- which a path or query string could
    spoof (e.g. a URL to ``evil.example`` with ``vlm-scheduling-service``
    somewhere in its path).
    """
    host = getattr(url, "host", None)
    if host is None:
        host = urlsplit(str(url)).hostname
    return host.lower() if host else None


def _destination_allowed(url: Any) -> bool:
    host = _extract_host(url)
    if host is None:
        return False
    return any(
        fnmatch.fnmatchcase(host, pattern.lower())
        for pattern in _allowed_destination_hosts
    )


def _find_pair(pairs: list[tuple[str, str]]) -> str | None:
    for key, value in pairs:
        if key.lower() == SOURCE_HEADER.lower():
            return value
    return None


# A caller-supplied value (including the source_headers() escape hatch, which
# legitimately equals what we'd inject ourselves) must never be touched, on
# any hop or destination -- but that value is indistinguishable from our own
# by content alone, since both are the same string. So "did *we* put this
# header there" has to be tracked out of band, not inferred from the value.
# Each library carries a hop's headers into the next redirect differently,
# which is why the tracking mechanism below differs per library; the policy
# itself (never touch a caller's value; add when allowed and absent; remove
# our own value when the destination is no longer allowed) is the same one
# in all three. Client/AsyncClient set the same INJECTED_EXTENSION_KEY marker
# (see client.py) so their own native header injection cooperates correctly
# with this scoping/origin-forwarding logic when instrument() is also active.

_EXTENSION_KEY = INJECTED_EXTENSION_KEY


def _inject_aiohttp_request(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """``aiohttp.ClientRequest.__init__`` -- ``ClientSession._request`` builds a
    fresh ``ClientRequest`` for the initial dispatch and again, with the
    redirect's URL, for every subsequent hop, all inside one ``while`` loop,
    always from the *original*, unmodified ``headers`` local variable (our own
    injection below only ever materializes a new pairs list passed into that
    one call, never mutates it) -- so a header present at any hop can only be
    something the caller put there, and no provenance tracking is needed here.

    Passing a list of pairs rather than a dict keeps any duplicate headers the
    caller supplied, and is also the one injection vehicle ``aioresponses`` can
    see, which a ``TraceConfig``-based approach cannot offer.
    """
    url = args[1] if len(args) > 1 else kwargs.get("url")
    headers = kwargs.get("headers")
    pairs = list(headers.items()) if hasattr(headers, "items") else list(headers or [])

    existing = _find_pair(pairs)
    if existing is None:
        source = _effective_source()
        if source and _destination_allowed(url):
            pairs.append((SOURCE_HEADER, source))

    # Always pass the materialized pairs onward. If `headers` was a one-shot
    # iterable, inspecting it above consumed the original even when nothing
    # about it needs to change.
    kwargs["headers"] = pairs

    return wrapped(*args, **kwargs)


_requests_dispatch_state: contextvars.ContextVar[dict[int, dict[str, bool]] | None] = (
    contextvars.ContextVar("visionai_requests_dispatch_state", default=None)
)


def _inject_requests_send(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """``requests.Session.send`` -- called for the initial dispatch and, via
    ``resolve_redirects``, recursively again for every redirect hop, nested
    inside this same (not-yet-returned) call. ``PreparedRequest.copy()`` carries
    a hop's headers *and* its ``hooks`` object (by reference, not copied) into
    the next one, so ``id(request.hooks)`` is stable across every hop of one
    redirect chain and distinct for any unrelated nested ``send()`` -- e.g. a
    response hook that fires (still inside this call) and makes its own,
    independent request. Keying state by that id, inside a per-context dict,
    means a hook's own call gets fresh state instead of incorrectly inheriting
    "we_injected" from the chain it happened to fire during.
    """
    request = args[0] if args else kwargs.get("request")
    if request is None:
        return wrapped(*args, **kwargs)

    states = _requests_dispatch_state.get()
    token = None
    if states is None:
        states = {}
        token = _requests_dispatch_state.set(states)

    chain_key = id(request.hooks)
    is_new_chain = chain_key not in states
    state = states.setdefault(chain_key, {"we_injected": False})

    try:
        source = _effective_source()
        existing = request.headers.get(SOURCE_HEADER)

        if existing is not None and not state["we_injected"]:
            pass  # caller-supplied (this hop or an earlier one); never touched
        elif source and _destination_allowed(request.url):
            request.headers[SOURCE_HEADER] = source
            state["we_injected"] = True
        elif existing is not None:
            del request.headers[SOURCE_HEADER]
            state["we_injected"] = False

        return wrapped(*args, **kwargs)
    finally:
        if is_new_chain:
            states.pop(chain_key, None)
        if token is not None:
            _requests_dispatch_state.reset(token)


def _apply_httpx_scoping(request: Any) -> None:
    """httpx: ``request.extensions`` is passed *by reference* into the next
    hop's ``Request`` by ``_build_redirect_request``, so it survives exactly
    the hops of one chain and nothing else -- the same role
    ``_requests_dispatch_state`` plays via a contextvar, except carried on the
    request object itself, since httpx's redirect loop calls
    ``_send_single_request`` repeatedly at the same call-stack depth (not
    nested like requests' recursive ``Session.send``), so a contextvar reset on
    return wouldn't survive between hops the way it does there.
    """
    source = _effective_source()
    existing = request.headers.get(SOURCE_HEADER)
    we_injected = request.extensions.get(_EXTENSION_KEY, False)

    if existing is not None and not we_injected:
        return  # caller-supplied (this hop or an earlier one); never touched

    if source and _destination_allowed(request.url):
        request.headers[SOURCE_HEADER] = source
        request.extensions[_EXTENSION_KEY] = True
    elif existing is not None:
        del request.headers[SOURCE_HEADER]
        request.extensions[_EXTENSION_KEY] = False


def _inject_httpx_send(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """``httpx.Client._send_single_request`` -- unlike ``Client.send``, this runs
    once per hop: the redirect loop in ``_send_handling_redirects`` calls it
    again for each redirect with a freshly-built request."""
    request = args[0] if args else kwargs.get("request")
    if request is not None:
        _apply_httpx_scoping(request)
    return wrapped(*args, **kwargs)


async def _inject_httpx_send_async(
    wrapped: Any, instance: Any, args: Any, kwargs: Any
) -> Any:
    """``httpx.AsyncClient._send_single_request`` -- the async twin of the above."""
    request = args[0] if args else kwargs.get("request")
    if request is not None:
        _apply_httpx_scoping(request)
    return await wrapped(*args, **kwargs)


# (module, attribute path, wrapper). Missing modules are skipped: requests and
# aiohttp are optional extras, httpx is a core dependency.
_TARGETS = (
    ("requests", "Session.send", _inject_requests_send),
    ("httpx", "Client._send_single_request", _inject_httpx_send),
    ("httpx", "AsyncClient._send_single_request", _inject_httpx_send_async),
    ("aiohttp", "ClientRequest.__init__", _inject_aiohttp_request),
)


def instrument(allowed_destination_hosts: list[str] | None = None) -> None:
    """Wrap the installed HTTP clients' request-dispatch methods. Safe to call
    more than once: the wrapping itself only happens the first time (calling it
    again would double-wrap the same methods). Omitting
    ``allowed_destination_hosts`` on a later call leaves whatever is already
    configured untouched (a no-op for scoping) -- only passing an explicit
    list updates it; there is no way to omit the argument and mean "reset to
    the default", since that can't be distinguished from "I have nothing new
    to say about this."

    ``allowed_destination_hosts`` gates which destinations receive
    ``X-Request-Source``; hostnames are matched via ``fnmatch`` glob patterns
    against the parsed URL host (case-insensitive), so ``*.svc.cluster.local``
    or ``*-backend.svc.cluster.local`` work as expected. Defaults to
    :data:`_DEFAULT_ALLOWED_DESTINATION_HOSTS` on the first call when omitted
    -- there is no "match everything" option; a service that needs a broader
    or different set must name it explicitly.
    """
    global _instrumented, _allowed_destination_hosts

    if allowed_destination_hosts is not None:
        _allowed_destination_hosts = tuple(allowed_destination_hosts)
    elif not _instrumented:
        _allowed_destination_hosts = _DEFAULT_ALLOWED_DESTINATION_HOSTS

    if _instrumented:
        return

    for module, target, wrapper in _TARGETS:
        try:
            wrapt.wrap_function_wrapper(module, target, wrapper)
        except ImportError:
            continue  # module not installed -- expected, it's an optional extra
        except AttributeError:
            warnings.warn(
                f"visionai_sdk_python: could not patch {module}.{target} for "
                f"{SOURCE_HEADER} attribution -- the target may have moved or "
                "been renamed upstream. Outbound calls through this library "
                "will not carry the header.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

    _instrumented = True


def uninstrument() -> None:
    """Restore the original clients. Mainly for test isolation."""
    global _instrumented
    if not _instrumented:
        return

    for module, target, _ in _TARGETS:
        try:
            obj = __import__(module)
            class_name, attr = target.split(".")
            cls = getattr(obj, class_name)
        except (ImportError, AttributeError):
            continue

        # vars() rather than getattr(): looking the attribute up normally would
        # run the descriptor protocol and hand back a *bound* wrapper, whose
        # __wrapped__ is not the plain function we need to put back. Duck-type on
        # __wrapped__ rather than isinstance(ObjectProxy) — under wrapt 2.x
        # FunctionWrapper is not an ObjectProxy instance.
        original = getattr(vars(cls).get(attr), "__wrapped__", None)
        if original is not None:
            setattr(cls, attr, original)

    _instrumented = False
