"""
Tool representation for the Twilio Agent Connect.

Inspired by OpenAI's function_schema approach from openai-agents-python (MIT License).
Injection pattern inspired by LangChain's InjectedToolArg system (MIT License).
"""

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, TypeAdapter

from tac import get_logger

if TYPE_CHECKING:
    # Typing-only soft dep: `openai-agents` is optional at runtime. The ignore
    # silences strict-mode downstream mypy when the package isn't installed
    # (and is a no-op when it is — `unused-ignore` covers both environments).
    from agents import FunctionTool  # type: ignore[import-not-found,unused-ignore]

logger = get_logger(__name__)


# Marker class for injected tool arguments
class InjectedToolArg:
    """
    Marker class for tool arguments that are injected at runtime.

    Tool arguments annotated with this class are not included in the tool
    schema sent to language models and are instead injected during execution.

    Inspired by LangChain's InjectedToolArg pattern.

    Example:
        @function_tool()
        def my_tool(
            user_input: str,
            client: Annotated[MyClient, InjectedToolArg]
        ) -> str:
            # client is injected, not visible to LLM
            return client.process(user_input)
    """

    pass


@dataclass
class TACTool:
    """
    Represents a tool/function that can be used with LLMs.

    Similar to OpenAI's FuncSchema, this captures function metadata
    for LLM tool integration. Supports runtime injection of dependencies
    that are hidden from the LLM schema.
    """

    name: str
    description: str
    params_json_schema: dict[str, object]
    _raw_implementation: Callable[..., object] = field(repr=False)
    _injected_args: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _injected_param_types: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _cached_callable: Callable[..., Awaitable[object]] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def implementation(self) -> Callable[..., Awaitable[object]]:
        """
        Get a clean callable with only non-injected parameters in its signature.

        This property automatically returns the right callable for LLM SDK introspection.
        The returned callable has only non-injected parameters in its signature while
        automatically handling dependency injection when called.

        Returns an async callable since TAC is async-first.

        Returns:
            An async callable with clean signature that can be inspected by any LLM SDK

        Example:
            # Pass to LLM SDK - it will introspect the clean signature
            sdk.add_tool(tool.implementation)
        """
        # Return cached callable if available and injection config hasn't changed
        if self._cached_callable is not None:
            return self._cached_callable

        properties = self.params_json_schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        non_injected_params: list[str] = list(properties.keys()) if properties else []

        sig = inspect.signature(self._raw_implementation)
        type_hints = get_type_hints(self._raw_implementation, include_extras=True)

        new_params: list[inspect.Parameter] = []
        for param_name in non_injected_params:
            param = sig.parameters.get(param_name)
            if param:
                param_type = type_hints.get(param_name)
                if param_type is None:
                    param_annotation = (
                        param.annotation if param.annotation != inspect.Parameter.empty else str
                    )
                    param_type = param_annotation
                origin = get_origin(param_type)
                if origin is Annotated:
                    args = get_args(param_type)
                    param_type = args[0] if args else param_type

                default_val = (
                    param.default
                    if param.default != inspect.Parameter.empty
                    else inspect.Parameter.empty
                )
                new_param = inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_val,
                    annotation=param_type,
                )
                new_params.append(new_param)

        new_sig = inspect.Signature(parameters=list(new_params))

        async def tool_callable(*args: object, **kwargs: object) -> object:
            """Clean async callable that forwards to TACTool.__call__()"""
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            call_kwargs: dict[str, object] = dict(bound.arguments)
            return await self(**call_kwargs)

        tool_callable.__name__ = self.name
        tool_callable.__doc__ = self.description
        # Assign __signature__ only if supported
        try:
            tool_callable.__signature__ = new_sig  # type: ignore[attr-defined]
        except Exception:
            pass

        self._cached_callable = tool_callable
        return tool_callable

    def configure_injection(self, **kwargs: object) -> "TACTool":
        """
        Configure values to be injected at runtime when the tool is called.

        These values correspond to parameters marked with InjectedToolArg
        annotations and will be automatically supplied when the tool executes.

        Validates that provided values match the expected types from the
        function signature using Pydantic TypeAdapter for robust validation
        of all Python type annotations including generics, Pydantic models,
        Literal types, and complex unions.

        Args:
            **kwargs: Mapping of parameter names to values to inject

        Returns:
            Self for method chaining

        Raises:
            TypeError: If a provided value doesn't match the expected type
            ValueError: If an unknown parameter name is provided

        Warning:
            Do not directly mutate _injected_args. Always use configure_injection()
            to ensure proper cache invalidation and type validation.

        Example:
            tool.configure_injection(client=conversation_memory_client, config=tac_config)
        """
        for param_name, value in kwargs.items():
            if param_name not in self._injected_param_types:
                raise ValueError(
                    f"Unknown injected parameter '{param_name}'. "
                    f"Expected one of: {list(self._injected_param_types.keys())}"
                )
            expected_type = self._injected_param_types[param_name]

            try:
                try:
                    is_pydantic_model = isinstance(expected_type, type) and issubclass(
                        expected_type, BaseModel
                    )
                except TypeError:
                    is_pydantic_model = False

                adapter: TypeAdapter[object]
                if is_pydantic_model:
                    adapter = TypeAdapter(expected_type)
                else:
                    adapter = TypeAdapter(expected_type, config={"arbitrary_types_allowed": True})

                adapter.validate_python(value)
            except Exception as pydantic_error:
                type_str = str(expected_type)
                if hasattr(expected_type, "__name__"):
                    type_str = expected_type.__name__
                elif hasattr(expected_type, "__origin__"):
                    type_str = str(expected_type)

                raise TypeError(
                    f"Type mismatch for parameter '{param_name}': "
                    f"expected {type_str}, got {type(value).__name__}. "
                    f"Validation error: {pydantic_error}"
                ) from pydantic_error

        self._injected_args.update(kwargs)
        self._cached_callable = None
        return self

    async def __call__(self, **kwargs: object) -> object:
        """
        Call the tool with the given arguments, automatically injecting
        configured dependencies.

        Handles both sync and async implementations transparently.

        Args:
            **kwargs: Arguments provided by the LLM or caller

        Returns:
            Result from the tool's implementation
        """
        # Merge LLM-provided args with injected args
        all_args: dict[str, object] = {**self._injected_args, **kwargs}

        if inspect.iscoroutinefunction(self._raw_implementation):
            result = await self._raw_implementation(**all_args)
        else:
            result = self._raw_implementation(**all_args)
        return result

    def to_openai_format(self) -> dict[str, object]:
        """
        Get tool schema in OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function format
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_json_schema,
            },
        }

    def to_realtime_format(self) -> dict[str, object]:
        """
        Get tool schema in OpenAI Realtime API function-calling format.

        Unlike ``to_openai_format`` (Chat Completions, which nests the schema
        under a ``"function"`` key), Realtime's ``session.tools`` expects the
        fields flat on the tool object.

        Returns:
            Dictionary in Realtime tool format
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params_json_schema,
        }

    def to_anthropic_format(self) -> dict[str, object]:
        """
        Get tool schema in Anthropic tool calling format.

        Returns:
            Dictionary in Anthropic tool format
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.params_json_schema,
        }

    def to_openai_agents_sdk_tool(self) -> "FunctionTool":
        """
        Convert this tool to an OpenAI Agents SDK ``FunctionTool`` instance.

        Unlike ``to_openai_format`` and ``to_anthropic_format`` (which return
        plain dicts consumed by HTTP APIs), the OpenAI Agents SDK dispatches
        on tool *class*, so this returns a live ``FunctionTool`` object with
        an ``on_invoke`` closure that calls this tool and JSON-encodes the
        result.

        Requires the ``openai-agents`` package:

            pip install openai-agents

        Returns:
            A ``FunctionTool`` ready to pass to ``Agent(tools=[...])``.
        """
        try:
            from agents import FunctionTool
        except ImportError as e:
            raise ImportError(
                "to_openai_agents_sdk_tool() requires the openai-agents package. "
                "Install with: pip install openai-agents"
            ) from e

        def _json_default(obj: object) -> object:
            # Tool implementations may return Pydantic models (e.g. the
            # knowledge tool returns list[KnowledgeChunkResult]), which the
            # stdlib json encoder can't handle on its own.
            if isinstance(obj, BaseModel):
                # mode="json" coerces datetime/UUID/Decimal to JSON primitives.
                return obj.model_dump(mode="json", by_alias=True, exclude_none=True)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        async def on_invoke(_ctx: object, args_json: str) -> str:
            args = json.loads(args_json) if args_json else {}
            logger.debug("TOOL | Called", tool_name=self.name)
            result = await self(**args)
            return json.dumps(result, default=_json_default)

        return FunctionTool(
            name=self.name,
            description=self.description,
            params_json_schema=self.params_json_schema,
            on_invoke_tool=on_invoke,
            # Disable strict mode: Agents SDK's strict_json_schema rejects
            # some of the JSON Schema features TAC emits (e.g. unions).
            strict_json_schema=False,
        )

    def to_json(self) -> str:
        """Convert tool to JSON string (OpenAI format by default)."""
        return json.dumps(self.to_openai_format(), indent=2)


def _is_injected_arg_type(param_type: object) -> bool:
    """
    Check if a type annotation indicates an injected argument.

    Looks for Annotated[T, InjectedToolArg] in the annotation metadata.

    Args:
        param_type: Type annotation to check

    Returns:
        True if the parameter should be injected at runtime
    """
    origin: object = get_origin(param_type)

    if origin is Annotated:
        metadata = get_args(param_type)[1:]
        for arg in metadata:
            try:
                if isinstance(arg, type) and issubclass(arg, InjectedToolArg):
                    return True
            except Exception:
                pass
            if arg is not None and not isinstance(arg, type):
                if isinstance(arg, InjectedToolArg):
                    return True

    return False


def _extract_injected_param_types(
    func: Callable[..., object],
) -> dict[str, object]:
    """
    Extract types of parameters marked with InjectedToolArg from function signature.

    Args:
        func: Function to extract injected parameter types from

    Returns:
        Dictionary mapping injected parameter names to their types (including generic types)
    """
    sig = inspect.signature(func)
    type_hints: dict[str, object] = get_type_hints(func, include_extras=True)

    injected_types: dict[str, object] = {}

    for param_name, _param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type: object = type_hints.get(param_name, str)

        if _is_injected_arg_type(param_type):
            origin = get_origin(param_type)
            if origin is Annotated:
                args = get_args(param_type)
                # Extract the base type from Annotated[T, InjectedToolArg]
                # This preserves generic types like list[str], dict[str, Any], etc.
                base_type = args[0] if args else str
                injected_types[param_name] = base_type
            else:
                if isinstance(param_type, type):
                    injected_types[param_name] = param_type
                else:
                    injected_types[param_name] = type(param_type)

    return injected_types


def _extract_schema_from_function(func: Callable[..., object]) -> dict[str, object]:
    """
    Extract JSON schema from function signature and type hints.

    Filters out parameters marked with InjectedToolArg annotations,
    as these are injected at runtime and should not be exposed to LLMs.

    Inspired by OpenAI's function_schema approach and LangChain's injection pattern.

    Args:
        func: Function to extract schema from

    Returns:
        JSON schema dictionary (without injected parameters)
    """
    sig = inspect.signature(func)
    type_hints: dict[str, object] = get_type_hints(func, include_extras=True)

    properties: dict[str, object] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type: object = type_hints.get(param_name, str)

        if _is_injected_arg_type(param_type):
            continue

        prop_schema: dict[str, object] = _type_to_json_schema(param_type)

        properties[param_name] = prop_schema

        default_val = getattr(param, "default", inspect.Parameter.empty)
        if default_val == inspect.Parameter.empty and not _is_optional(param_type):
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _type_to_json_schema(param_type: object) -> dict[str, object]:
    """
    Convert Python type to JSON schema using Pydantic TypeAdapter.

    This leverages Pydantic's robust type system to handle all type annotations
    including Literal, Enum, complex Unions, nested models, and more.

    For Optional[T] types (both Optional[T] and T | None), we unwrap to the
    base type since optionality is tracked separately via the 'required' array
    in the parent schema.
    """
    import types

    origin = get_origin(param_type)
    args = get_args(param_type)

    # Handle Optional[T] (both typing.Union and types.UnionType for T | None)
    if origin is Union or origin is types.UnionType:
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            param_type = non_none_type

    try:
        adapter: TypeAdapter[object] = TypeAdapter(param_type)
        schema = adapter.json_schema(mode="validation")

        # Remove Pydantic-specific metadata fields that aren't needed for LLM tools
        schema.pop("title", None)
        schema.pop("$defs", None)

        return schema
    except Exception:
        return {"type": "string"}


def _is_optional(param_type: object) -> bool:
    """Check if a type is Optional (Union with None)."""
    import types
    from typing import Union

    origin = get_origin(param_type)

    # Check both typing.Union (for backward compatibility) and types.UnionType (Python 3.10+)
    if origin is Union or origin is types.UnionType:
        args = get_args(param_type)
        return type(None) in args
    return False


def function_tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., object]], TACTool]:
    """
    Decorator to create a TAC tool from a function.

    Similar to OpenAI's function_tool decorator approach.

    Args:
        name: Optional name override (defaults to function name)
        description: Optional description override (defaults to docstring)

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., object]) -> TACTool:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()

        if not tool_description:
            raise ValueError(f"Function {func.__name__} must have a docstring or description")

        schema = _extract_schema_from_function(func)

        tool = TACTool(
            name=tool_name,
            description=tool_description,
            params_json_schema=schema,
            _raw_implementation=func,
        )

        tool._injected_param_types = _extract_injected_param_types(func)

        return tool

    return decorator


def create_tool(
    name: str,
    description: str,
    params_json_schema: dict[str, object],
    implementation: Callable[..., object],
) -> TACTool:
    """
    Create a TAC tool manually with explicit schema.

    Args:
        name: The name of the tool/function
        description: Description of what the tool does
        params_json_schema: JSON Schema for the tool's parameters
        implementation: Function that implements the tool's logic

    Returns:
        TACTool instance
    """
    tool = TACTool(
        name=name,
        description=description,
        params_json_schema=params_json_schema,
        _raw_implementation=implementation,
    )

    tool._injected_param_types = _extract_injected_param_types(implementation)

    return tool
