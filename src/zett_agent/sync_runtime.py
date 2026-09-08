"""Blocking bridges that keep async resources on one persistent event loop."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from concurrent.futures import Future
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, copy_context
from functools import wraps
from queue import Queue
from threading import Event, Lock, Thread, get_ident
from typing import Any, Self, overload

_callback_loop: ContextVar[asyncio.AbstractEventLoop | None] = ContextVar("zett_sync_callback_loop", default=None)


class SyncRuntime(AbstractContextManager):
    """Own one background event loop for all blocking calls in a scope.

    Reuse this scope when calling a provider, Agent, or extension repeatedly.
    Calls preserve context variables and propagate original exceptions. Different
    calling threads may share a runtime. Closing cancels streams, awaits their
    cleanup, and shuts down the loop and its worker pool. Objects passed into
    this runtime remain caller-owned: close their SDK/database resources first.

    Examples:
        Call any asynchronous function without writing an async entry point::

            with SyncRuntime() as runtime:
                reply = runtime.call(agent.run, "Hello")
                with runtime.stream(agent.stream, "Continue") as events:
                    for event in events:
                        print(event.type)

    .. warning::
        Do not move already-used asynchronous SDK clients between event loops.
        Construct and use them within the same runtime. A blocking call from
        the runtime's own loop is rejected instead of deadlocking.
    """

    _loop: asyncio.AbstractEventLoop
    _thread_id: int
    _thread: Thread | None

    def __init__(self) -> None:
        self._lock = Lock()
        self._closed = False
        self._ready = Event()
        self._streams: set[SyncStream[Any]] = set()
        self._thread = Thread(target=self._serve, name="zett-agent-sync", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _serve(self) -> None:
        with asyncio.Runner() as runner:
            self._loop = runner.get_loop()
            self._thread_id = get_ident()
            self._ready.set()
            self._loop.run_forever()

    @classmethod
    def _borrow(cls) -> SyncRuntime:
        """Borrow the running loop for a synchronous hook executing in a worker."""
        runtime = object.__new__(cls)
        runtime._loop = asyncio.get_running_loop()
        runtime._thread_id = get_ident()
        runtime._lock = Lock()
        runtime._closed = False
        runtime._streams = set()
        runtime._thread = None
        return runtime

    def _check_thread(self) -> None:
        if get_ident() == self._thread_id:
            raise RuntimeError("A synchronous call cannot block its own event loop; use await instead")

    def _check_close(self) -> None:
        self._check_thread()
        if _callback_loop.get() is self._loop:
            raise RuntimeError(
                "A synchronous callback cannot close its own runtime or stream; return to the caller first"
            )

    def _submit[T](self, function: Callable[[], Coroutine[Any, Any, T]]) -> Future[T]:
        self._check_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("SyncRuntime is closed")
            coroutine = function()
            try:
                return asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            except BaseException:
                coroutine.close()
                raise

    def call[**P, T](self, function: Callable[P, T | Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Invoke a callable on the owned loop and wait for its final value.

        Both ordinary functions and awaitables are accepted. Use :meth:`stream`
        for async iterators and :meth:`wrap` for a complete service object.
        KeyboardInterrupt cancels the submitted operation.
        """

        async def invoke() -> T:
            result = function(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        future = self._submit(invoke)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def stream[**P, T](
        self, function: Callable[P, AsyncIterator[T]], /, *args: P.args, **kwargs: P.kwargs
    ) -> SyncStream[T]:
        """Create a lazy, closeable iterator with one owner task for its lifetime.

        Each next() requests exactly one event; no model output is eagerly
        buffered. Use a with block when breaking early. Closing cancels an
        in-flight next() and waits for the producer's finally blocks.
        """
        self._check_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("SyncRuntime is closed")
            stream = SyncStream(self, lambda: function(*args, **kwargs))
            self._streams.add(stream)
        return stream

    def wrap[T](self, value: T) -> SyncObject[T]:
        """Expose an object's async methods as blocking methods on this runtime."""
        return SyncObject(value, runtime=self)

    def context[T](self, value: AbstractAsyncContextManager[T]) -> SyncContext[T]:
        """Enter and exit an async context manager in the same owner task."""
        return SyncContext(self, value)

    def close(self) -> None:
        """Cancel owned streams and join the loop thread; repeated calls are safe."""
        self._check_close()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            streams = tuple(self._streams)
        errors: list[BaseException] = []
        try:
            for stream in streams:
                try:
                    stream.close()
                except BaseException as error:
                    errors.append(error)
        finally:
            if self._thread is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()
        if errors:
            raise BaseExceptionGroup("Synchronous stream cleanup failed", errors)

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("SyncRuntime is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SyncStream[T](Iterator[T], AbstractContextManager):
    """Pull async events synchronously, preserving one task and bounded memory.

    A stream has one consumer. Another thread may call close() to interrupt a
    blocked consumer. Exhaustion, errors, and close release the source iterator.
    Do not rely on garbage collection to close an abandoned stream.
    """

    def __init__(self, runtime: SyncRuntime, factory: Callable[[], AsyncIterator[T]]) -> None:
        self._runtime = runtime
        self._factory = factory
        self._lock = Lock()
        self._consumer = Lock()
        self._closed = False
        self._started = False
        self._done = Event()
        self._error: BaseException | None = None
        self._error_delivered = False
        self._replies: Queue[tuple[bool, Any]] = Queue()
        self._task: asyncio.Task[None] | None = None
        self._demand: asyncio.Queue[None] | None = None

    async def _produce(self) -> None:
        self._task = asyncio.current_task()
        self._demand = asyncio.Queue()
        terminal: BaseException | None = None
        try:
            if self._closed:
                return
            # The first next() starts production; subsequent steps need a permit.
            source = self._factory()
            try:
                while True:
                    try:
                        item = await anext(source)
                    except StopAsyncIteration:
                        break
                    self._replies.put((True, item))
                    await self._demand.get()
            finally:
                close = getattr(source, "aclose", None)
                if close is not None:
                    await close()
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            terminal = error
        finally:
            self._error = terminal
            self._replies.put((False, terminal))
            self._done.set()
            with self._runtime._lock:
                self._runtime._streams.discard(self)

    def __next__(self) -> T:
        self._runtime._check_thread()
        if not self._consumer.acquire(blocking=False):
            raise RuntimeError("A SyncStream supports only one consumer at a time")
        try:
            with self._lock:
                if self._closed:
                    raise StopIteration
                if not self._started:
                    self._runtime._submit(self._produce)
                    self._started = True
                elif not self._done.is_set():
                    self._runtime._loop.call_soon_threadsafe(self._request_next)
            success, value = self._replies.get()
            if not success:
                self._closed = True
                if value is not None:
                    self._error_delivered = True
                    raise value
                raise StopIteration
            return value
        except (KeyboardInterrupt, SystemExit):
            self.close()
            raise
        finally:
            self._consumer.release()

    def _request_next(self) -> None:
        if self._demand is not None and not self._done.is_set():
            self._demand.put_nowait(None)

    def close(self) -> None:
        """Stop consumption and wait until the async source has been closed."""
        self._runtime._check_close()
        with self._lock:
            self._closed = True
            started = self._started
        if started and not self._done.is_set():

            def cancel() -> None:
                if self._task is not None and not self._task.done():
                    self._task.cancel()

            self._runtime._loop.call_soon_threadsafe(cancel)
            self._done.wait()
        with self._runtime._lock:
            self._runtime._streams.discard(self)
        if self._error is not None and not self._error_delivered:
            self._error_delivered = True
            raise self._error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SyncObject[T](AbstractContextManager):
    """A blocking view of an existing Agent, provider, tool, context, or extension.

    The original object remains available as ``wrapped``. Attributes retain
    their original types; async method results are awaited and async iterators
    become :class:`SyncStream`. Methods and tools can be called normally.
    Leaving this scope closes only a runtime created by this view. Explicitly
    call provider.aclose() on the view to release a caller-owned SDK client.

    Examples:
        Use any provider's ordinary API synchronously::

            with provider.sync() as blocking:
                with blocking.stream(request) as events:
                    for event in events:
                        print(event)
                blocking.aclose()
    """

    wrapped: T
    runtime: SyncRuntime
    _owns_runtime: bool

    def __init__(self, wrapped: T, *, runtime: SyncRuntime | None = None) -> None:
        object.__setattr__(self, "wrapped", wrapped)
        object.__setattr__(self, "runtime", runtime if runtime is not None else SyncRuntime())
        object.__setattr__(self, "_owns_runtime", runtime is None)

    @overload
    def _invoke[**ParametersT, ResultT](
        self,
        function: Callable[ParametersT, AsyncIterator[ResultT] | Awaitable[AsyncIterator[ResultT]]],
        *args: ParametersT.args,
        **kwargs: ParametersT.kwargs,
    ) -> SyncStream[ResultT]: ...

    @overload
    def _invoke[**ParametersT, ResultT](
        self,
        function: Callable[
            ParametersT, AbstractAsyncContextManager[ResultT] | Awaitable[AbstractAsyncContextManager[ResultT]]
        ],
        *args: ParametersT.args,
        **kwargs: ParametersT.kwargs,
    ) -> SyncContext[ResultT]: ...

    @overload
    def _invoke[**ParametersT, ResultT](
        self,
        function: Callable[ParametersT, Awaitable[ResultT]],
        *args: ParametersT.args,
        **kwargs: ParametersT.kwargs,
    ) -> ResultT: ...

    @overload
    def _invoke[**ParametersT, ResultT](
        self,
        function: Callable[ParametersT, ResultT],
        *args: ParametersT.args,
        **kwargs: ParametersT.kwargs,
    ) -> ResultT: ...

    def _invoke[**ParametersT, ResultT](
        self,
        function: Callable[ParametersT, ResultT],
        *args: ParametersT.args,
        **kwargs: ParametersT.kwargs,
    ) -> object:
        """Preserve callable parameters and expose its blocking result type.

        Awaitables are resolved, asynchronous iterators become SyncStream,
        and asynchronous context managers become SyncContext. Specific
        overloads precede the ordinary-result fallback to describe these
        conversions instead of claiming that the return type is unchanged.
        """
        if inspect.isasyncgenfunction(function):
            return self.runtime.stream(function, *args, **kwargs)
        value = self.runtime.call(function, *args, **kwargs)
        if isinstance(value, AsyncIterator):
            return self.runtime.stream(lambda: value)
        if isinstance(value, AbstractAsyncContextManager):
            return self.runtime.context(value)
        return value

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.wrapped, name)
        if not callable(value):
            return value

        @wraps(value)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            return self._invoke(value, *args, **kwargs)

        return invoke

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.wrapped, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke(self.wrapped, *args, **kwargs)

    def __hash__(self) -> int:
        return hash(self.wrapped)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SyncObject) and self.wrapped is other.wrapped

    def __enter__(self) -> Self:
        self.runtime.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._owns_runtime:
            self.runtime.close()


class SyncMethodsMixin:
    """Expose all asynchronous instance methods through a blocking view."""

    __slots__ = ()

    def sync(self, *, runtime: SyncRuntime | None = None) -> SyncObject[Self]:
        """Return a context-managed synchronous view; optionally share a runtime.

        Examples:
            Use a model or tool from a plain function::

                with read_file.sync() as read:
                    result = read({"path": "README.md"})
        """
        return SyncObject(self, runtime=runtime)


class SyncContext[T](AbstractContextManager):
    """Preserve async context-manager task ownership and exception suppression."""

    def __init__(self, runtime: SyncRuntime, source: AbstractAsyncContextManager[T]) -> None:
        self._source = source
        self._exception: tuple[Any, Any, Any] = (None, None, None)
        self._events = runtime.stream(self._lifecycle)

    async def _lifecycle(self) -> AsyncIterator[Any]:
        value = await self._source.__aenter__()
        try:
            yield value
        except BaseException:
            if not await self._source.__aexit__(*sys.exc_info()):
                raise
        else:
            yield await self._source.__aexit__(*self._exception)

    def __enter__(self) -> T:
        return next(self._events)

    def __exit__(self, *exc: Any) -> bool:
        self._exception = exc
        try:
            return bool(next(self._events))
        finally:
            self._events.close()


async def _worker(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Wait for a running synchronous callback before releasing its context."""

    def invoke() -> Any:
        token = _callback_loop.set(loop)
        try:
            return function(*args, **kwargs)
        finally:
            _callback_loop.reset(token)

    loop = asyncio.get_running_loop()
    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise


def sync_hook(function: Callable[..., Any], *, context: bool = False, streaming: bool = False) -> Callable[..., Any]:
    """Adapt a synchronous override without blocking the Agent's event loop."""
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        return function

    def arguments(args: tuple[Any, ...], runtime: SyncRuntime) -> tuple[Any, ...]:
        if context and len(args) > 1:
            return (args[0], runtime.wrap(args[1]), *args[2:])
        return args

    if inspect.isgeneratorfunction(function) or streaming:

        @wraps(function)
        async def events(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            runtime = SyncRuntime._borrow()
            sentinel = object()
            worker_context = copy_context()
            worker_context.run(_callback_loop.set, asyncio.get_running_loop())
            source = await _worker(worker_context.run, function, *arguments(args, runtime), **kwargs)
            source = iter(()) if source is None else iter(source)
            try:
                while True:
                    item = await _worker(worker_context.run, next, source, sentinel)
                    if item is sentinel:
                        break
                    yield item
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    await _worker(worker_context.run, close)

        return events

    @wraps(function)
    async def invoke(*args: Any, **kwargs: Any) -> Any:
        runtime = SyncRuntime._borrow()
        result = await _worker(function, *arguments(args, runtime), **kwargs)
        return await result if inspect.isawaitable(result) else result

    return invoke
