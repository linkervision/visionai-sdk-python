"""Source attribution for outbound HTTP, with no call-site changes.

Call ``instrument()`` once at service startup, naming the destinations that
should receive the header::

    from visionai_sdk_python import instrumentation
    instrumentation.instrument(
        allowed_destination_hosts=["vlm-inference-server", "*.svc.cluster.local"],
    )

Every request through ``requests``, ``httpx`` (sync and async) and ``aiohttp``
then carries ``X-Request-Source`` -- including third-party code, module-level
one-shots (``requests.get()``), and clients built before ``instrument()`` ran,
since injection patches a request-dispatch method on the class rather than
``__init__``.

**Destination scoping.** ``allowed_destination_hosts`` is required on the
first call -- there is no default, and no "all destinations" mode. Guessing a
default that doesn't match a service's actual hosts would silently inject
nothing, indistinguishable from attribution simply not being wired up; a
service instead states explicitly which destinations should carry the header,
so a mismatch either fails immediately (see below) or fails loudly (the
process-lifetime warning below). Everything outside the list -- Stripe, an
external webhook, ... -- is untouched.

Re-checked on every hop, so a redirect leaving the allowlist stops carrying the
header instead of leaking it onward. If ``allowed_destination_hosts`` never
matches any destination, a ``RuntimeWarning`` fires -- as soon as it's been at
least ``_EARLY_WARN_SECONDS`` since ``instrument()`` and another unmatched
request is dispatched, or at interpreter shutdown if that never happens --
"attribution has no data" and "the allowlist is wrong" would otherwise look
identical in production. The early check exists because a bare ``SIGTERM``
(the usual container shutdown signal) skips ``atexit`` hooks entirely, so
relying on shutdown alone would mean a killed container never surfaces a
misconfigured allowlist at all.

This warning tracks whether a *destination* matched the allowlist, not
whether a header was actually sent, so it can't catch a correct allowlist
paired with an unset ``VISIONAI_SERVICE_SOURCE`` (e.g. left out of Helm
values) -- every call still reaches an allowed destination, so nothing ever
looks unmatched, even though no header goes out. ``instrument()`` checks for
that directly and warns once, at call time, if the env var is unset.

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

import atexit
import contextvars
import fnmatch
import os
import time
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
    SOURCE_ENV_VAR,
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
    "SUGGESTED_ALLOWED_DESTINATION_HOSTS",
]

_instrumented = False

# A starting point services can pass explicitly -- not applied automatically.
# There is no default and no "match everything" mode: guessing one that
# doesn't match a service's real hosts would silently inject nothing, which
# looks identical to attribution never having been wired up at all.
SUGGESTED_ALLOWED_DESTINATION_HOSTS: tuple[str, ...] = (
    "vlm-inference-server",
    "vlm-scheduling-service",
    "*-backend.svc.cluster.local",
    "*.svc.cluster.local",
)

_allowed_destination_hosts: tuple[str, ...] = ()

# Process-lifetime "did we ever actually inject the header" tracking -- catches
# an allowlist that's syntactically fine but never matches anything the
# service actually calls, which otherwise looks identical to "attribution has
# no data for some other reason" until someone thinks to check the allowlist.
_ever_matched_destination = False
_saw_any_dispatch = False
_lifetime_check_registered = False
_warned_never_matched = False

# atexit doesn't run on a bare SIGTERM (the usual container shutdown signal),
# so the warning also fires early -- from the next unmatched dispatch, not a
# background timer -- once this much time has passed since instrument() with
# still no match, instead of only ever being delivered at a shutdown that may
# never reach atexit.
_EARLY_WARN_SECONDS = 30.0
_instrumented_at: float | None = None


def _note_dispatch(matched: bool) -> None:
    # Trade-off, not a bug: a service running with PYTHONWARNINGS=error (or
    # -W error) turns this into a RuntimeWarning raised straight at whichever
    # request happens to be the first unmatched dispatch after
    # _EARLY_WARN_SECONDS -- _warned_never_matched below is set before the
    # warning fires, so only that one request pays for it, but which request
    # that is isn't deterministic. Acceptable since warnings-as-errors is the
    # caller's own choice, but worth remembering if this symptom is ever
    # reported.
    global _ever_matched_destination, _saw_any_dispatch
    _saw_any_dispatch = True
    if matched:
        _ever_matched_destination = True
        return
    if (
        _instrumented_at is not None
        and time.monotonic() - _instrumented_at >= _EARLY_WARN_SECONDS
    ):
        _warn_if_never_matched()


def _warn_if_never_matched() -> None:
    """Registered via atexit; also called early (see ``_note_dispatch``) and
    directly by tests, since atexit hooks are unreliable to trigger and
    observe from within a test process. ``_warned_never_matched`` guards
    against firing twice for the same misconfiguration -- once early and then
    again at shutdown."""
    global _warned_never_matched
    if _warned_never_matched:
        return
    if _saw_any_dispatch and not _ever_matched_destination:
        _warned_never_matched = True
        warnings.warn(
            f"visionai_sdk_python: instrument() was called and requests were "
            f"made, but allowed_destination_hosts {_allowed_destination_hosts!r} "
            f"never matched any destination -- {SOURCE_HEADER} was never sent. "
            "This usually means the allowlist doesn't match the service's "
            "actual outbound hosts.",
            RuntimeWarning,
            stacklevel=2,
        )


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
    matched = host is not None and any(
        fnmatch.fnmatchcase(host, pattern.lower())
        for pattern in _allowed_destination_hosts
    )
    _note_dispatch(matched)
    return matched


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
        matched = _destination_allowed(url)
        source = _effective_source()
        if source and matched:
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

    Only mutates a ``.copy()`` of the *first* hop's request, never the object
    the caller passed in directly -- mutating it in place left an observable
    side effect on the caller's own ``PreparedRequest``, and one that outlives
    a single ``send()``: reusing that same object for a later, unrelated call
    (mutate ``.url``, call ``send()`` again) would see our earlier value as
    "already there", indistinguishable from something the caller set
    themselves.

    Every later hop's request, though, is one ``resolve_redirects()`` built
    itself (via its own ``req.copy()``) and will never be seen by any other
    code -- and critically, mutating *that* copy is the only way our decision
    is visible to ``resolve_redirects()``'s own chaining. It keeps its own
    ``req`` variable across iterations of its redirect loop and copies from
    *that* to build each next hop, entirely independent of whatever object we
    hand to ``wrapped()``; a ``.copy()`` here, like the first hop's, would only
    ever affect what that one hop actually sends over the wire, invisible to
    ``resolve_redirects()`` -- so hop 3's request would still be built from
    hop 1's original header, un-stripped, regardless of what hop 2 decided.
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
        existing = request.headers.get(SOURCE_HEADER)

        if existing is not None and not state["we_injected"]:
            pass  # caller-supplied (this hop or an earlier one); never touched
        else:
            matched = _destination_allowed(request.url)
            source = _effective_source()
            if is_new_chain:
                if source and matched:
                    request = request.copy()
                    request.headers[SOURCE_HEADER] = source
                    state["we_injected"] = True
                elif existing is not None:
                    request = request.copy()
                    del request.headers[SOURCE_HEADER]
                    state["we_injected"] = False
            else:
                if source and matched:
                    request.headers[SOURCE_HEADER] = source
                    state["we_injected"] = True
                elif existing is not None:
                    del request.headers[SOURCE_HEADER]
                    state["we_injected"] = False

        if args:
            args = (request, *args[1:])
        else:
            kwargs["request"] = request
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
    existing = request.headers.get(SOURCE_HEADER)
    we_injected = request.extensions.get(_EXTENSION_KEY, False)

    if existing is not None and not we_injected:
        return  # caller-supplied (this hop or an earlier one); never touched

    matched = _destination_allowed(request.url)
    source = _effective_source()
    if source and matched:
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
    or ``*-backend.svc.cluster.local`` work as expected.

    Required on the first call -- there is no default and no "match
    everything" option. See :data:`SUGGESTED_ALLOWED_DESTINATION_HOSTS` for a
    starting point to pass explicitly.

    Raises:
        ValueError: if this is the first call and ``allowed_destination_hosts``
            was omitted, or if it was given but names no actual destination
            (an empty list, one containing only blank strings, or a bare
            string instead of a list of patterns -- a bare string is itself
            iterable character-by-character, which ``tuple()`` would
            otherwise accept silently, turning a typo like
            ``allowed_destination_hosts="*.example.com"`` (missing the
            brackets) into a set of single-character patterns including a
            lone ``"*"``, which matches every host).
    """
    global _instrumented, _allowed_destination_hosts, _lifetime_check_registered
    global _instrumented_at

    if allowed_destination_hosts is not None:
        if isinstance(allowed_destination_hosts, (str, bytes)):
            raise ValueError(
                "allowed_destination_hosts must be a list of hostname "
                "patterns, not a bare string -- a string is itself iterable "
                "character-by-character, which would otherwise be accepted "
                "silently as a set of single-character patterns (e.g. a "
                "lone '*', matching every host). Wrap it in a list: "
                f"allowed_destination_hosts=[{allowed_destination_hosts!r}]."
            )
        hosts = tuple(allowed_destination_hosts)
        if not any(host and host.strip() for host in hosts):
            raise ValueError(
                "allowed_destination_hosts must name at least one destination "
                "-- an empty list (or one containing only blank strings) "
                "silently matches nothing, indistinguishable from attribution "
                "never being wired up (see SUGGESTED_ALLOWED_DESTINATION_HOSTS "
                "for a starting point)."
            )
        _allowed_destination_hosts = hosts
    elif not _instrumented:
        raise ValueError(
            "instrument() requires allowed_destination_hosts on the first "
            "call -- there is no default, so name the destinations that "
            "should receive X-Request-Source explicitly (see "
            "SUGGESTED_ALLOWED_DESTINATION_HOSTS for a starting point)."
        )

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

    if not _lifetime_check_registered:
        atexit.register(_warn_if_never_matched)
        _lifetime_check_registered = True

    _raw_source = os.environ.get(SOURCE_ENV_VAR)
    if _raw_source is None or not _raw_source.strip():
        # Checked against the raw env var, not _source_value() -- a value
        # that's set but fails validation already gets its own warning (from
        # _validate(), on every request), and reporting that case as "unset"
        # here too would misdescribe it. But _validate() itself treats an
        # empty/whitespace-only value as absent and returns None *without*
        # warning (it's not "malformed", just blank), so that case needs to
        # be caught here too -- otherwise it's silent on both sides, and
        # empty is the likely real-world shape of "left out of Helm values"
        # (a blank ConfigMap key, `value: ""`), not a rare edge case.
        warnings.warn(
            f"visionai_sdk_python: instrument() was called but {SOURCE_ENV_VAR} "
            f"is unset or empty -- outbound requests will carry no "
            f"{SOURCE_HEADER} at all (unless every call happens inside a "
            "current_origin() scope that already has a value). A correct "
            "allowed_destination_hosts won't surface this: destination "
            "matching, and therefore the allowlist-never-matched warning, "
            "doesn't depend on whether a header was actually injected. This "
            "is usually a missing environment variable (e.g. left out of "
            "Helm values), not intentional.",
            RuntimeWarning,
            stacklevel=2,
        )

    _instrumented_at = time.monotonic()
    _instrumented = True


def uninstrument() -> None:
    """Restore the original clients. Mainly for test isolation."""
    global _instrumented, _allowed_destination_hosts
    global _ever_matched_destination, _saw_any_dispatch
    global _warned_never_matched, _instrumented_at
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
    _allowed_destination_hosts = ()
    _ever_matched_destination = False
    _saw_any_dispatch = False
    _warned_never_matched = False
    _instrumented_at = None
